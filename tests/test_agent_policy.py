import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from cross_episode_sim.agent_policy.interface import validate
from cross_episode_sim.agent_policy.history import catalog,frames,public_catalog

class AgentTests(unittest.TestCase):
    def test_memory_disabled(self):
        import jsonschema
        with self.assertRaises(jsonschema.ValidationError):
            validate({'name':'history_frame','arguments':{'video_id':'h0','seconds':[0]}},'none')
    def test_nonfinite(self):
        with self.assertRaises(Exception):validate({'name':'move_base','arguments':{'forward':float('nan'),'left':0,'yaw':0,'seconds':1}},'none')
    def test_no_unlisted_args(self):
        with self.assertRaises(Exception):validate({'name':'observe','arguments':{'object_pose':1}},'none')
    def test_no_oracle_tool(self):
        with self.assertRaises(Exception):validate({'name':'pick','arguments':{}},'all')
    def test_catalog_no_paths(self):
        self.assertEqual(public_catalog([{'video_id':'h0','source':'private','path':'private','seconds':1}]),[{'video_id':'h0','seconds':1}])
    def test_no_arbitrary_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):frames([], '/etc/passwd',[0],tmp)
    def test_training_manifest_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'train';p.mkdir()
            (p/'manifest.json').write_text(json.dumps({'episodes':[{'folder':'../test/source'}]}))
            with self.assertRaises(ValueError):catalog(p)
    def test_valid_wait(self):
        a={'name':'wait','arguments':{'seconds':.2}}
        self.assertEqual(validate(a,'none'),a)

if __name__=='__main__':unittest.main()
