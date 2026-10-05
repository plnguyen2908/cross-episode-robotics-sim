"""Standalone thin-object pickup checkpoint; shared fallback lives in edge_access."""
import traceback
from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
from cross_episode_sim.manipulation.edge_access import (EdgeAccess, edge_frame, outline_interval, side_annotations,
    object_bodies, tabletop_bounds, edge_candidates, install_thin_scene, thin_annotations)

class ThinEdgePickTest(CrossRoomManipulation):
    video_filename='thin_edge_pick.mp4'

    def __init__(self,args,selection):
        self.edge_pushing=False
        super().__init__(args,selection)
        self.source=args.source_table;self.support_bids=self.table_bids[self.source]
        self.report.update(task='thin object: direct side grasp or physical edge push, side pick and hold',edge_push_used=False)

    def run_test(self):
        success=False
        try:
            self.tick(1.);self.initial_object_pose=self.bread_pose().copy()
            self.review_phase='CHECK DIRECT THIN OBJECT PICK'
            self.pick_payload()
            success=bool(self.report.get('grasp_qualified') and self.holding_loaf)
        except Exception as exc:
            self.report.update(error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
        finally:
            self.report.update(success=success,scope='flat thin object on clear rectangular tabletop; physical side pickup and two-second hold; no placement')
            self.finish_run_outputs(success)
        return 0 if success else 1
