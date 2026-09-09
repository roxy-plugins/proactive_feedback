from datetime import datetime,timezone,timedelta
from pathlib import Path
import json,sqlite3
import pytest
from test_plugin import module,_plugin_context
from bus.events_lifecycle import TurnCommitted
from session.manager import SessionManager


def setup_messages(tmp_path):
    manager=SessionManager(tmp_path)
    first=manager.get_or_create('mobile:source')
    first.add_message('assistant','值得讨论',proactive=True,delivery_id='delivery')
    manager.save(first)
    source_id=first.messages[0]['id']
    second=manager.get_or_create('mobile:discussion')
    second.metadata['discussion_source']={'session_id':first.key,'message_id':source_id}
    second.add_message('assistant','引用的 Roxy 来信：值得讨论',source_refs=[{'kind':'discussion_source','session_id':first.key,'message_id':source_id}])
    second.add_message('user','能展开讲讲吗？')
    second.add_message('assistant','这条发现涉及以下内容。')
    manager.save(second)
    return manager,source_id,second


@pytest.mark.asyncio
async def test_discussion_is_attributed_to_original_message_and_retry_is_noop(tmp_path):
    manager,source_id,discussion=setup_messages(tmp_path)
    plugin=module.ProactiveFeedbackPlugin();plugin.context=_plugin_context(tmp_path);plugin.activate()
    class Embedding:
        async def embed_batch(self,texts):return [[1.0,0.0]]*len(texts)
    plugin._embedder=Embedding()
    event=TurnCommitted(session_key=discussion.key,channel='mobile',chat_id='discussion',input_message='能展开讲讲吗？',
        persisted_user_message='能展开讲讲吗？',assistant_response='这条发现涉及以下内容。',tools_used=[],
        persisted_user_message_id=str(discussion.messages[1]['id']),assistant_message_id=str(discussion.messages[2]['id']))
    try:
        await plugin._process(event);await plugin._process(event)
        with sqlite3.connect(plugin._db_path) as db:
            rows=db.execute('SELECT id,session_key,proactive_message_id,matched_by FROM proactive_feedback_events').fetchall()
        assert len(rows)==1 and rows[0][1:]==(discussion.key,source_id,'discussion_source')
        assert rows[0][0]==1
    finally:await plugin.terminate();manager.close()


def test_identity_first_short_quote_and_scope_rejection(tmp_path):
    manager,source_id,discussion=setup_messages(tmp_path)
    try:
        with sqlite3.connect(manager.db_path) as db:
            db.row_factory=sqlite3.Row
            pair=module.latest_turn_messages(db,session_key=discussion.key,user_content='wrong repeated text',assistant_content='wrong',
                    user_message_id=str(discussion.messages[1]['id']),assistant_message_id=str(discussion.messages[2]['id']))
            assert pair is not None
            target,kind=module.referenced_proactive(db,session_key=discussion.key,user=pair[0])
            assert target.id==source_id and kind=='discussion_source'
        original=manager.get_existing('mobile:source');original.metadata['private']=True;manager.save(original)
        with sqlite3.connect(manager.db_path) as db:
            db.row_factory=sqlite3.Row
            target,kind=module.referenced_proactive(db,session_key=discussion.key,user=pair[0])
            assert target is None and kind=='invalid_reference'
    finally:manager.close()
