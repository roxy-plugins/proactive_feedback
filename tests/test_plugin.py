from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path

import pytest

from agent.plugins.context import PluginContext, PluginKVStore
from agent.plugins.scope import PluginScope, ScopedEventBus
from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted


def _load_plugin_module():
    path = Path(__file__).parents[1] / "plugin.py"
    spec = importlib.util.spec_from_file_location(
        "test_proactive_feedback_plugin",
        path,
        submodule_search_locations=[str(path.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plugin_context(tmp_path: Path) -> PluginContext:
    scope = PluginScope("proactive_feedback")
    return PluginContext(
        event_bus=ScopedEventBus(EventBus(), scope),
        tool_registry=None,
        plugin_id="proactive_feedback",
        plugin_dir=tmp_path,
        data_dir=tmp_path,
        kv_store=PluginKVStore(tmp_path / ".kv.json"),
        workspace=tmp_path,
        scope=scope,
        _can_start_tasks=lambda: True,
    )


module = _load_plugin_module()
ProactiveFeedbackPlugin = module.ProactiveFeedbackPlugin
FeedbackEvent = module.FeedbackEvent


def _write_sessions_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE sessions (key TEXT PRIMARY KEY, metadata TEXT)")
        conn.execute("INSERT INTO sessions VALUES ('web:test', '{}')")
        _ = conn.execute("""
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                extra TEXT,
                ts TEXT NOT NULL
            )
            """)
        _ = conn.executemany(
            """
            INSERT INTO messages(id, session_key, seq, role, content, extra, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "p1",
                    "web:test",
                    1,
                    "assistant",
                    "AI Agent Runtime 趋势",
                    json.dumps({"proactive": True}),
                    "2026-08-18T10:00:00+00:00",
                ),
                (
                    "u1",
                    "web:test",
                    2,
                    "user",
                    "请继续讲 AI Agent Runtime",
                    None,
                    "2026-08-18T10:01:00+00:00",
                ),
                (
                    "a1",
                    "web:test",
                    3,
                    "assistant",
                    "下面继续解释 Runtime 的分层。",
                    None,
                    "2026-08-18T10:01:01+00:00",
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_proactive_feedback_summary_empty(tmp_path: Path) -> None:
    plugin = ProactiveFeedbackPlugin()
    scope = PluginScope("proactive_feedback")
    plugin.context = PluginContext(
        event_bus=ScopedEventBus(EventBus(), scope),
        tool_registry=None,
        plugin_id="proactive_feedback",
        plugin_dir=tmp_path,
        data_dir=tmp_path,
        kv_store=PluginKVStore(tmp_path / ".kv.json"),
        workspace=tmp_path,
        scope=scope,
        _can_start_tasks=lambda: True,
    )
    plugin.activate()
    try:
        summary = await plugin.get_summary(None)
    finally:
        await plugin.terminate()
        assert await scope.aclose() == []
    assert summary["total"] == 0


@pytest.mark.asyncio
async def test_turn_committed_produces_feedback_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实 TurnCommitted fanout 应驱动 worker 写入反馈库。"""

    _write_sessions_db(tmp_path / "sessions.db")
    plugin = ProactiveFeedbackPlugin()
    plugin.context = _plugin_context(tmp_path)

    async def fake_score_followup(**kwargs: object) -> SimpleNamespace:
        candidates = kwargs["candidates"]
        assert isinstance(candidates, list)
        return SimpleNamespace(
            proactive=candidates[0],
            pa_score=0.9,
            pua_score=0.8,
            matched_by="recent_pua",
            feedback_type="topic_follow",
            confidence="high",
            reason="test_followup",
            candidate_count=len(candidates),
            lag_seconds=61,
        )

    monkeypatch.setattr(module, "score_followup", fake_score_followup)
    monkeypatch.setattr(
        plugin,
        "_get_embedder",
        lambda: SimpleNamespace(embed_batch=lambda texts: texts),
    )
    plugin.activate()
    try:
        db_path = tmp_path / "proactive_feedback" / "proactive_feedback.db"
        assert db_path.exists()
        event = TurnCommitted(
            session_key="web:test",
            channel="web",
            chat_id="test",
            input_message="请继续讲 AI Agent Runtime",
            persisted_user_message="请继续讲 AI Agent Runtime",
            assistant_response="下面继续解释 Runtime 的分层。",
            tools_used=[],
            turn_id="turn-1",
            timestamp=datetime(2026, 8, 18, tzinfo=timezone.utc),
        )
        await plugin.context.event_bus.fanout(event)
        await plugin._queue.join()
        with module.open_db(db_path) as conn:
            row = conn.execute(
                "SELECT feedback_type, user_message_id, proactive_message_id "
                "FROM proactive_feedback_events"
            ).fetchone()
        assert row is not None
        assert row["feedback_type"] == "topic_follow"
        assert row["user_message_id"] == "u1"
        assert row["proactive_message_id"] == "p1"
    finally:
        await plugin.terminate()


@pytest.mark.asyncio
async def test_activate_initializes_schema(tmp_path: Path) -> None:
    plugin = ProactiveFeedbackPlugin()
    plugin.context = _plugin_context(tmp_path)
    plugin.activate()
    try:
        db_path = tmp_path / "proactive_feedback" / "proactive_feedback.db"
        assert db_path.exists()
        with module.open_db(db_path) as conn:
            assert module.schema_version(conn) == 1
            columns = {
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(proactive_feedback_events)"
                ).fetchall()
            }
        assert "feedback_type" in columns
        assert "user_message_id" in columns
    finally:
        await plugin.terminate()


def test_recorded_event_matches_runtime_shape() -> None:
    feedback = FeedbackEvent(
        session_key="telegram:1",
        user_message_id="u1",
        assistant_message_id="a1",
        proactive_message_id="p1",
        feedback_type="topic_follow",
        confidence="high",
        pa_score=0.9,
        pua_score=0.8,
        lag_seconds=12,
        candidate_count=2,
        matched_by="recent_pua",
        reason="matched",
    )

    event = module._recorded_event(7, feedback)

    assert event.event_id == 7
    assert event.session_key == "telegram:1"
    assert event.user_message_id == "u1"
    assert event.assistant_message_id == "a1"
    assert event.proactive_message_id == "p1"
    assert event.feedback_type == "topic_follow"
    assert event.confidence == "high"
    assert event.pua_score == 0.8
    assert event.lag_seconds == 12
    assert event.matched_by == "recent_pua"


def test_get_embedder_uses_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[Path] = []

    def fake_build_embedder(root: Path) -> object:
        seen.append(root)
        return object()

    plugin = ProactiveFeedbackPlugin()
    plugin.context = _plugin_context(tmp_path)
    plugin._workspace = tmp_path
    plugin._embedder = None
    monkeypatch.setattr(module, "_build_embedder", fake_build_embedder)

    embedder = plugin._get_embedder()

    assert embedder is plugin._embedder
    assert seen == [tmp_path]


def test_mobile_contribution_declares_dashboard() -> None:
    contribution = ProactiveFeedbackPlugin.mobile_ui()

    assert contribution.module == "mobile_panel.js"
    assert contribution.stylesheet == "mobile_panel.css"
    assert contribution.navigation is not None
    assert contribution.navigation.label == "主动反馈"


def test_mobile_feedback_projection_reuses_dashboard_reader(tmp_path: Path) -> None:
    plugin = ProactiveFeedbackPlugin()
    plugin.context = _plugin_context(tmp_path)
    sink = module.open_db(tmp_path / "proactive_feedback" / "proactive_feedback.db")
    try:
        module.insert_feedback(
            sink,
            FeedbackEvent(
                session_key="mobile:test",
                user_message_id="u-mobile",
                assistant_message_id="a-mobile",
                proactive_message_id="p-mobile",
                feedback_type="explicit_quote",
                confidence="gold",
                pa_score=1.0,
                pua_score=None,
                lag_seconds=8,
                candidate_count=1,
                matched_by="quote",
                reason="explicit_quote",
            ),
        )
    finally:
        sink.close()

    overview = plugin.mobile_ui_query(
        "feedback.overview",
        {},
        session_id=None,
        turn_id=None,
    )
    page = plugin.mobile_ui_query(
        "feedback.events",
        {"page": 1, "page_size": 30, "feedback_type": "explicit_quote"},
        session_id=None,
        turn_id=None,
    )

    assert overview["total"] == 1
    assert overview["follow_rate"] == 1.0
    assert page["total"] == 1
    assert page["items"][0]["feedback_type"] == "explicit_quote"


def test_mobile_feedback_projection_rejects_unknown_filter(tmp_path: Path) -> None:
    plugin = ProactiveFeedbackPlugin()
    plugin.context = _plugin_context(tmp_path)

    with pytest.raises(ValueError, match="feedback_type 不受支持"):
        plugin.mobile_ui_query(
            "feedback.events",
            {"feedback_type": "invented"},
            session_id=None,
            turn_id=None,
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"page": True}, "page 必须"),
        ({"page_size": 51}, "page_size 必须"),
    ],
)
def test_mobile_feedback_projection_rejects_invalid_page(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    plugin = ProactiveFeedbackPlugin()
    plugin.context = _plugin_context(tmp_path)

    with pytest.raises(ValueError, match=message):
        plugin.mobile_ui_query(
            "feedback.events",
            payload,
            session_id=None,
            turn_id=None,
        )


def test_mobile_feedback_projection_rejects_unknown_method(tmp_path: Path) -> None:
    plugin = ProactiveFeedbackPlugin()
    plugin.context = _plugin_context(tmp_path)

    with pytest.raises(ValueError, match="未知 proactive_feedback 移动方法"):
        plugin.mobile_ui_query(
            "feedback.delete",
            {},
            session_id=None,
            turn_id=None,
        )
