"""Conservative room footprints from observed wall columns, never house metadata."""
import numpy as np
from scipy.ndimage import binary_closing,binary_dilation,binary_fill_holes,label,distance_transform_edt,binary_opening
from .room_exploration_policy import RoomEvidence,PortalEvidence


class ObservedRoomTopology:
    def __init__(self,minimum_area=3.):
        self.minimum_area=minimum_area;self.rooms={};self.previous={};self.portals={};self.events=[];self.seams=[];self.active=None

    def update(self,grid,position):
        resolution=grid.config.resolution
        # Tall columns propose partitions only; never modify navigation evidence.
        z=grid.config.min_obstacle_height+(np.arange(grid.evidence.shape[2])+.5)*grid.config.height_resolution
        walls=((grid.evidence[:,:,z>=1.3]>=2).sum(axis=2)>=3)
        for center,normal,width in self.seams:
            tangent=np.array([-normal[1],normal[0]])
            points=center+np.linspace(-width,width,1+int(2*width/resolution))[:,None]*tangent
            cells=grid.world_to_cell(points);valid=(cells[:,0]>=0)&(cells[:,1]>=0)&(cells[:,0]<walls.shape[1])&(cells[:,1]<walls.shape[0])
            walls[cells[valid,1],cells[valid,0]]=True
        closed=binary_closing(walls,iterations=max(1,round(.12/resolution)))
        filled=binary_fill_holes(closed)
        interiors=filled & ~closed
        cell=np.floor((np.asarray(position[:2])-getattr(grid,'origin',np.zeros(2)))/resolution).astype(int)
        inside=(0<=cell[1]<interiors.shape[0] and 0<=cell[0]<interiors.shape[1] and interiors[cell[1],cell[0]])
        if interiors.sum()*resolution**2<self.minimum_area or not inside:
            interiors |= supported_wall_enclosure(walls,position,grid)
        # The fallback is an enclosure proposal, not permission to erase
        # observed internal walls or the traversed-door partition.
        interiors &= ~closed
        # Open doorways should not merge otherwise separate room interiors.
        # Separate large interior cores at narrow observed necks, then assign
        # the remaining interior to those cores without deleting floor area.
        components,n=partition_interiors(interiors,resolution,self.minimum_area)
        floor=grid.floor_views>=grid.config.minimum_floor_views
        obstacles=(grid.evidence>=2).any(axis=2)
        current=None;matched=set()
        for i in range(1,n+1):
            mask=components==i
            if mask.sum()*resolution**2<self.minimum_area:continue
            rr,cc=np.where(mask)
            xy=grid.cell_to_world(np.c_[cc,rr])
            world=np.floor(xy/resolution+1e-7).astype(int)
            footprint=frozenset(map(tuple,world))
            overlaps=[(len(footprint & old['footprint'])/max(1,min(len(footprint),len(old['footprint']))),key) for key,old in self.rooms.items() if key not in matched]
            match=max(overlaps,default=(0,None))
            key=match[1] if match[0]>.5 else (self.active if self.active is not None and not self.rooms[self.active]['footprint'] else f'room_{len(self.rooms):03d}')
            matched.add(key)
            previous=self.previous.get(key,frozenset())
            agreement=len(footprint & previous)/max(1,len(footprint | previous))
            eligible=~obstacles[rr,cc]
            navigable=frozenset(map(tuple,world[eligible]))
            observed=frozenset(map(tuple,world[eligible & floor[rr,cc]]))
            evidence=RoomEvidence(key,navigable,observed,agreement>.9)
            self.accept_room_observation(key,footprint,evidence)
            cell=tuple(np.floor(np.asarray(position[:2])/resolution+1e-7).astype(int))
            if cell in self.rooms[key]['footprint']:current=key
        # Never declare 100% of the discovered free patch to be a complete room.
        if current is None:
            cell=tuple(np.floor(np.asarray(position[:2])/resolution+1e-7).astype(int))
            current=next((key for key,value in self.rooms.items() if cell in value['footprint']),None)
        if current is None:
            if self.active is None:
                self.active=f'room_{len(self.rooms):03d}'
                self.rooms[self.active]=dict(footprint=frozenset(),evidence=RoomEvidence(self.active,frozenset(),frozenset(),False))
            current=self.active
        self.active=current
        return current

    def accept_room_observation(self,key,footprint,evidence):
        """Keep an established enclosure while its replacement is unstable.

        Door motion may change the contour for one or more frames. That does
        not erase previously observed coverage. A repeated, supported new
        enclosure replaces the old one, including genuinely expanded area.
        Navigation occupancy is independent and is never frozen by this method.
        """
        old=self.rooms.get(key)
        retain=old is not None and old['evidence'].boundary_supported and not evidence.boundary_supported
        self.previous[key]=footprint
        if retain:
            if not old.get('boundary_refresh_pending',False):
                self.events.append(dict(event='retain_supported_room',room=key,coverage=old['evidence'].coverage))
            old['boundary_refresh_pending']=True
            return
        self.rooms[key]=dict(footprint=footprint,evidence=evidence,boundary_refresh_pending=False)

    def enter_after_crossing(self,source,current,known_destination=None):
        """A known portal returns to its existing node, never a phantom room."""
        if known_destination in self.rooms and known_destination!=source:
            self.active=known_destination
            return known_destination
        if current in self.rooms and current!=source:
            self.active=current
            return current
        return self.enter_unobserved_room()

    def enter_unobserved_room(self):
        """Register a provisional node after an observed doorway crossing.

        No footprint or coverage is invented; both require subsequent sensing.
        """
        key=f'room_{len(self.rooms):03d}'
        self.rooms[key]=dict(footprint=frozenset(),evidence=RoomEvidence(key,frozenset(),frozenset(),False))
        self.active=key
        return key

    def associate_portals(self,tracks,resolution):
        """Associate only unique, supported room footprints on opposite sides.

        Nearby rooms on the same side or ambiguous corners remain unknown.
        This can identify a closed shortcut after an alternative route was used.
        """
        for track in tracks:
            if track.get('destination') is not None or track['observations']<2:continue
            near={}
            for key,value in self.rooms.items():
                if not value['evidence'].boundary_supported or not value['footprint']:continue
                xy=(np.array(list(value['footprint']))+.5)*resolution
                delta=xy-track['position'][:2];distance=np.linalg.norm(delta,axis=1)
                if distance.min()<.7:
                    vector=delta[distance<distance.min()+.15].mean(axis=0)
                    if np.linalg.norm(vector)>.05:near[key]=vector/np.linalg.norm(vector)
            source=track.get('source')
            if source not in near:continue
            others=[key for key in near if key!=source]
            if len(others)==1 and near[source]@near[others[0]]<-.5:
                track['destination']=others[0]
                track['association']='opposite_supported_observed_footprints'

    def uncovered_points(self,points,resolution,target=.9):
        """Novel room space, excluding boundary noise around covered footprints.

        The margin is only an exploration utility filter. Occupancy, floor
        evidence and collision clearance are untouched.
        """
        covered=set()
        for value in self.rooms.values():
            c=value['evidence'].coverage
            if c is not None and c>=target:covered.update(value['footprint'])
        points=np.asarray(points)
        if not covered:return np.zeros(len(points),bool)
        from scipy.spatial import cKDTree
        centers=(np.asarray(list(covered))+.5)*resolution
        distance,_=cKDTree(centers).query(points)
        return distance>.3+resolution/2

    def frontier_leaves_covered_rooms(self,points,resolution,target=.9):
        priorities=self.uncovered_points(points,resolution,target)
        return bool(len(priorities) and priorities.mean()>.5)

    def evidence(self):return [r['evidence'] for r in self.rooms.values()]

    def summary(self):
        return [dict(room=r.room,coverage=r.coverage,boundary_supported=r.boundary_supported,
                     boundary_refresh_pending=self.rooms[r.room].get('boundary_refresh_pending',False),
                     estimated_cells=len(r.estimated_navigable),observed_cells=len(r.observed_navigable)) for r in self.evidence()]


def supported_wall_enclosure(walls,position,grid):
    """Propose a rectangular enclosure only with four long observed wall runs.

    Bridges occlusion for coverage estimation only, not navigation. Nonrectangular
    rooms retain the contour estimator; this fallback refuses unsupported sides.
    """
    import cv2
    resolution=grid.config.resolution
    lines=cv2.HoughLinesP(walls.astype(np.uint8),1,np.pi/180,threshold=10,
                         minLineLength=max(10,int(1.3/resolution)),maxLineGap=max(2,int(.2/resolution)))
    if lines is None:return np.zeros_like(walls)
    lines=lines[:,0,:];delta=lines[:,2:]-lines[:,:2]
    lengths=np.linalg.norm(delta,axis=1);angles=np.arctan2(delta[:,1],delta[:,0])
    angle=angles[np.argmax(lengths)]%(np.pi/2)
    R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    centers=grid.cell_to_world((lines[:,:2]+lines[:,2:])/2)@R
    local=np.asarray(position[:2])@R
    horizontal=abs(np.sin(angles-angle))<.08;vertical=abs(np.cos(angles-angle))<.08
    bounds=[]
    for axis,keep in [(0,vertical),(1,horizontal)]:
        offsets=centers[keep,axis]
        low=offsets[offsets<local[axis]-.6];high=offsets[offsets>local[axis]+.6]
        if not len(low) or not len(high):return np.zeros_like(walls)
        bounds.append((low.max(),high.min()))
    lo=np.array([b[0] for b in bounds]);hi=np.array([b[1] for b in bounds])
    if np.any(hi-lo<1.5):return np.zeros_like(walls)
    rr,cc=np.where(walls);observed=grid.cell_to_world(np.c_[cc,rr])@R
    for axis in (0,1):
        other=1-axis
        for value in (lo[axis],hi[axis]):
            along=observed[(abs(observed[:,axis]-value)<.15)&(observed[:,other]>=lo[other])&(observed[:,other]<=hi[other]),other]
            bins=np.unique(np.floor((along-lo[other])/resolution+1e-7).astype(int))
            if len(bins)*resolution/(hi[other]-lo[other])<.6:return np.zeros_like(walls)
    yy,xx=np.indices(walls.shape);points=grid.cell_to_world(np.stack((xx,yy),-1))@R
    return ((points>lo+.1)&(points<hi-.1)).all(axis=-1)


def partition_interiors(interiors,resolution,minimum_area):
    from skimage.segmentation import watershed
    distance=distance_transform_edt(interiors)*resolution
    cores,n=label(distance>.8)
    markers=np.zeros_like(cores);count=0
    for i in range(1,n+1):
        mask=cores==i
        if mask.sum()*resolution**2>=minimum_area*.25:
            count+=1;markers[mask]=count
    if count<2:return label(interiors)
    # Retain isolated components too; no unobserved area is invented.
    components,n=label(interiors)
    for i in range(1,n+1):
        mask=components==i
        if not np.any(markers[mask]):
            count+=1;markers[mask]=count
    return watershed(-distance,markers,mask=interiors),count
