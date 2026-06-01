import argparse
import json
import os
import sys
import time
from PIL import ImageDraw
import copy
import numpy as np
import cv2

# Simulation
sys.path.append('../simulation')
from unity_simulator.comm_unity import (
    UnityCommunication,
    UnityEngineException,
    UnityCommunicationException,
)
from unity_simulator import utils_viz
from ros_utils import *
from utils_demo import *
from graph_utils import *

## ROS Service Calls
import rclpy
from rclpy.node import Node
from amrl_msgs.srv import (
    GetImageSrv,
    GetImageAtPoseSrv, 
    PickObjectSrv, 
    GetVisibleObjectsSrv,
    FindObjectSrv,
    SemanticObjectDetectionSrv,
    ChangeVirtualHomeGraphSrv,
    DetectVirtualHomeObjectSrv,
    OpenVirtualHomeObjectSrv,
)
from geometry_msgs.msg import Point

GetImageSrvResponse = GetImageSrv.Response
GetImageAtPoseSrvResponse = GetImageAtPoseSrv.Response
PickObjectSrvResponse = PickObjectSrv.Response
GetVisibleObjectsSrvResponse = GetVisibleObjectsSrv.Response
FindObjectSrvResponse = FindObjectSrv.Response
SemanticObjectDetectionSrvRequest = SemanticObjectDetectionSrv.Request
SemanticObjectDetectionSrvResponse = SemanticObjectDetectionSrv.Response
ChangeVirtualHomeGraphSrvResponse = ChangeVirtualHomeGraphSrv.Response
DetectVirtualHomeObjectSrvRequest = DetectVirtualHomeObjectSrv.Request
DetectVirtualHomeObjectSrvResponse = DetectVirtualHomeObjectSrv.Response
OpenVirtualHomeObjectSrvRequest = OpenVirtualHomeObjectSrv.Request
OpenVirtualHomeObjectSrvResponse = OpenVirtualHomeObjectSrv.Response


class _RospyShim:
    ServiceException = Exception

    def __init__(self):
        self._node = None
        self._services = []

    def init_node(self, name: str, anonymous: bool = True):
        if not rclpy.ok():
            rclpy.init()
        self._node = Node(name)

    def wait_for_service(self, name: str):
        return

    def ServiceProxy(self, name: str, srv_type):
        node = self._node
        if node is None:
            raise RuntimeError("ROS2 node is not initialized")
        client = node.create_client(srv_type, name)
        while not client.wait_for_service(timeout_sec=1.0):
            self.logwarn(f"Waiting for service: {name}")

        class _Proxy:
            def __call__(self_inner, request):
                future = client.call_async(request)
                rclpy.spin_until_future_complete(node, future)
                result = future.result()
                if result is None:
                    raise _RospyShim.ServiceException(f"Service call failed: {name}")
                return result

        return _Proxy()

    def Service(self, name: str, srv_type, handler):
        node = self._node
        if node is None:
            raise RuntimeError("ROS2 node is not initialized")

        def _callback(request, response):
            return handler(request)

        service = node.create_service(srv_type, name, _callback)
        self._services.append(service)
        return service

    def loginfo(self, msg: str):
        if self._node is None:
            print(msg)
            return
        self._node.get_logger().info(msg)

    def logwarn(self, msg: str):
        if self._node is None:
            print(msg)
            return
        self._node.get_logger().warning(msg)

    def logerr(self, msg: str):
        if self._node is None:
            print(msg)
            return
        self._node.get_logger().error(msg)

    def spin(self):
        if self._node is None:
            raise RuntimeError("ROS2 node is not initialized")
        try:
            rclpy.spin(self._node)
        finally:
            self._node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


rospy = _RospyShim()

comm = None
class_list = None
cameras_select = None
pano_camera_select = None
first_person_pano_camera_select = None
tall_pano_camera_select = None

# --- Pano observation snapshot cache (off unless --snapshot_obs) ----------
# When enabled, every successful scene_set / navigate snapshots the four
# pano modes (normal, depth, seg_class, seg_inst) into _snapshot_obs_cache.
# Subsequent observe / detect / find / pick reads consult the cache instead
# of calling comm.camera_image again, so RGB/depth/seg used by a single
# request are mutually pixel-consistent. Open invalidates; lazy populate
# on first miss. Off by default — service is bit-for-bit unchanged.
SNAPSHOT_OBS_MODES = ("normal", "depth", "seg_class", "seg_inst")
_snapshot_obs_enabled = False
_snapshot_obs_cache = None  # None | dict[str, list of np.ndarray]

# --- Long-range detect flag -----------------------------------------------
# When enabled, detection (_detect_instance and _detect_objects) extends its
# DEPTH_MAX from 2m to 5m, so the agent can perceive farther objects. Pick
# and open keep their own 2m gate: if the target instance's mean masked
# depth is farther than 2m, they fail immediately. Off by default —
# detection, pick, and open all behave bit-for-bit like before.
_long_range_detect_enabled = False
PICK_OPEN_DEPTH_MAX = 2.0

# When enabled, observe() saves a debug pano grid to ../../outputs/debug_observe.png.
_verbose_enabled = False


def parse_args():
    parser = argparse.ArgumentParser(description='Virtual Home ROS Service')
    parser.add_argument('--port', type=str, required=True, help='Port for Unity communication')
    parser.add_argument('--parallel', action='store_true', help='Namespace services as /moma_{port}/... for parallel runs')
    parser.add_argument('--snapshot_obs', action='store_true',
                        help='Cache pano camera images (normal/depth/seg_class/seg_inst) at '
                             'the character pose after each successful scene set / navigate. '
                             'observe/detect/find/pick read from the cache; open invalidates. '
                             'Default off — service behaves bit-for-bit like before.')
    parser.add_argument('--long_range_detect', action='store_true',
                        help='Extend detection DEPTH_MAX from 2m to 5m for _detect_instance '
                             'and _detect_objects. Pick and open keep their own 2m gate and '
                             'fail when the target instance is farther than 2m. Default off — '
                             'detection/pick/open behave bit-for-bit like before.')
    parser.add_argument('--verbose', action='store_true',
                        help='Save debug artifacts (e.g. observe() pano grid to '
                             '../../outputs/debug_observe.png). Default off.')
    # parser.add_argument("--graph_path", type=str, required=True, help="Path to the scene graph")
    return parser.parse_args()


def _camera_image_with_retry(camera_select, mode, attempts=2):
    """``comm.camera_image`` with one retry on a Unity communication failure.

    A wedged Unity render surfaces as ``UnityCommunicationException`` (the HTTP
    request read-times-out). An occasional GPU/render hiccup clears on a second
    try; a genuine renderer deadlock will not — the retry just confirms the
    sim is dead so the caller can fail fast (and the start_sims watchdog can
    restart it). Re-raises the last exception when every attempt fails."""
    global comm
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return comm.camera_image(camera_select, mode=mode)
        except UnityCommunicationException as e:
            last_exc = e
            tail = "retrying" if attempt < attempts else "giving up"
            rospy.logwarn(
                f"snapshot_obs: camera_image(mode={mode}) attempt "
                f"{attempt}/{attempts} failed ({e}); {tail}"
            )
    raise last_exc


def _snapshot_obs_populate() -> bool:
    """Fetch all four pano modes on the current pano_camera_select and replace
    the snapshot cache. Returns True only if all four fetches succeeded."""
    global _snapshot_obs_cache, comm, pano_camera_select
    if pano_camera_select is None:
        rospy.logwarn("snapshot_obs: pano_camera_select is None; skipping populate")
        return False
    new_cache = {}
    for mode in SNAPSHOT_OBS_MODES:
        try:
            ok, imgs = _camera_image_with_retry(pano_camera_select, mode)
        except UnityCommunicationException as e:
            rospy.logerr(
                f"snapshot_obs: camera_image(mode={mode}) failed after retry "
                f"({e}); cache cleared"
            )
            _snapshot_obs_cache = None
            return False
        if not ok:
            rospy.logwarn(f"snapshot_obs: camera_image(mode={mode}) failed; cache cleared")
            _snapshot_obs_cache = None
            return False
        new_cache[mode] = imgs
    _snapshot_obs_cache = new_cache
    rospy.loginfo(
        f"snapshot_obs: cached {len(new_cache['normal'])} pano frames x "
        f"{len(SNAPSHOT_OBS_MODES)} modes"
    )
    return True


def _snapshot_obs_invalidate() -> None:
    global _snapshot_obs_cache
    if _snapshot_obs_cache is not None:
        rospy.loginfo("snapshot_obs: cache invalidated")
    _snapshot_obs_cache = None


def _get_pano_images(mode: str):
    """Drop-in replacement for `comm.camera_image(pano_camera_select, mode=...)`.
    When --snapshot_obs is enabled, serves the requested mode from the cache,
    lazily populating all four modes on a miss. When the flag is off, forwards
    straight to comm.camera_image (preserving historical semantics).

    Returns (success: bool, imgs: list)."""
    global _snapshot_obs_enabled, _snapshot_obs_cache, comm, pano_camera_select
    if not _snapshot_obs_enabled:
        return comm.camera_image(pano_camera_select, mode=mode)
    if _snapshot_obs_cache is None:
        if not _snapshot_obs_populate() or _snapshot_obs_cache is None:
            return False, []
    return True, _snapshot_obs_cache[mode]


def get_moma_service_name(port: str, service: str, parallel: bool) -> str:
    if parallel:
        return f'/moma_{port}/{service}'
    return f'/moma/{service}'

### Helper Functions ###
def detect_objects_owlv2(query_image: Image, query_cls: str) -> SemanticObjectDetectionSrvResponse:
    """
    Detect objects by class.
    """
    rospy.wait_for_service("/owlv2/semantic_object_detection")
    try:
        detect_service = rospy.ServiceProxy("/owlv2/semantic_object_detection", SemanticObjectDetectionSrv)
        req = SemanticObjectDetectionSrvRequest()
        req.query_image = query_image
        req.query_text = query_cls
        response = detect_service(req)
        return response
    except rospy.ServiceException as e:
        print("Service call failed:", e)

def observe():
    global comm, pano_camera_select

    (ok_img, imgs) = _get_pano_images("normal")
    if ok_img and _verbose_enabled:
        view_pil = display_grid_img(imgs, nrows=2)
        debug_path = "../../outputs/debug_observe.png"
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        view_pil.save(debug_path)

    ros_images = []
    for img in imgs:
        ros_img = opencv_to_ros_image(img)
        ros_images.append(ros_img)

    return ros_images

### Handle Service Requests ###
def handle_navigate_request(req):
    global comm, pano_camera_select, first_person_pano_camera_select, tall_pano_camera_select
    try:
        x = req.x
        y = req.y
        z = req.z if req.z is not None and req.z > 0 else 0.0
        rospy.loginfo(f"Received navigate request: ({x}, {0}, {y})")
        
        success = comm.move_character(0, [x, 0, y])
        rospy.loginfo(f"Move character success: {success}")
        if not success:
            return GetImageAtPoseSrvResponse(success=False)
        if z > 0.3:
            pano_camera_select = copy.deepcopy(tall_pano_camera_select)
        else:
            pano_camera_select = copy.deepcopy(first_person_pano_camera_select)
        if _snapshot_obs_enabled:
            _snapshot_obs_populate()
        pano_images = observe()
        return GetImageAtPoseSrvResponse(success=success, pano_images=pano_images)
    except Exception as e:
        rospy.logerr(f"Error in navigate request: {e}")
        import traceback; traceback.print_exc()
        return GetImageAtPoseSrvResponse(success=False)


def handle_observe_request(req):
    global comm
    rospy.loginfo("Received observe request")
    
    ros_images = observe()
    return GetImageSrvResponse(
        image=ros_images[0], 
        pano_images=ros_images,
    )
    
def handle_visible_objects_request(req):
    global comm
    rospy.loginfo("Received visible objects request")
    
    unique_ids = set()
    for cam in pano_camera_select:
        _, visible_objects = comm.get_visible_objects(cam)
        for obj_id in visible_objects.keys():
            unique_ids.add(int(obj_id))
    
    success, graph = comm.environment_graph()
    nodes = extract_nodes_by_ids(graph["nodes"], unique_ids)
    
    # Format return values
    ids = [int(node["id"]) for node in nodes]
    classnames = [node["class_name"] for node in nodes]
    prefabnames = [node["prefab_name"] for node in nodes]

    return GetVisibleObjectsSrvResponse(
        ids=ids,
        classnames=classnames,
        prefabnames=prefabnames
    )
    
def find_target_node_id(query_text):
    # NOTE: As of the --snapshot_obs change, this function was intentionally
    # left untouched. It is stale: in current usage `handle_find_request` always
    # supplies a `ref_image` (taking the `_find_instance` path) and
    # `handle_pick_request` always supplies an `instance_id` (taking the
    # `_detect_instance` path), so this fallback is unreachable. It also
    # rotates the character via `comm.render_script([TurnRight])`, which would
    # require per-iteration cache invalidation to integrate cleanly with the
    # snapshot cache. If a future caller starts hitting this function with
    # --snapshot_obs enabled, plumb cache invalidation into the TurnRight
    # branch and switch the camera_image calls below to a cache-aware helper.
    depth_thresh = 3.0

    target_node_id = None

    for _ in range(6):
        ok_img, normal_imgs = comm.camera_image(cameras_select[2:3], mode="normal")
        ok_img, cls_imgs = comm.camera_image(cameras_select[2:3], mode="seg_class")
        ok_img, depth_imgs = comm.camera_image(cameras_select[2:3], mode="depth")
        normal_img = normal_imgs[0]
        cls_img = cls_imgs[0]
        
        depth_img = depth_imgs[0]
        if depth_img.ndim == 3 and depth_img.shape[2] == 4:
            depth_scalar_img = depth_img[..., 0]
        else:
            depth_scalar_img = depth_img
        valid_mask = (depth_scalar_img < depth_thresh)
        bgr_masked = cls_img[valid_mask]
        
        target_color = semantic_cls_to_bgr(query_text, class_list)
        match_mask = np.all(bgr_masked == target_color, axis=-1)
        display_grid_img(normal_imgs, nrows=1).save("../../outputs/debug.png")
        if np.any(match_mask):
            _, visible_objects = comm.get_visible_objects(cameras_select[2])
            for node_id, cls_name in visible_objects.items():
                if cls_name.lower() == query_text.lower():
                    target_node_id = int(node_id)
                    rospy.loginfo(f"Found target node ID: {target_node_id}")
                    return target_node_id 
    
        script = ["<char0> [TurnRight]", "<char0> [TurnRight]"]
        success, message = comm.render_script(script=script,
                                    processing_time_limit=30,
                                    find_solution=False,
                                    image_width=640,
                                    image_height=480,  
                                    skip_animation=True,
                                    recording=False,
                                    save_pose_data=False)
        if not success:
            rospy.logerr(f"Failed to turn character: {message}")
            return FindObjectSrvResponse(success=False, id=None)
        
    return target_node_id

def _get_query_text(txt: str) -> str:
    if "toy" in txt or "action figure" in txt or "transformer" in txt or "robot" in txt or "plush" in txt or "animal" in txt or "teddy" in txt or "train" in txt:
        return "toy"
    elif "magazine" in txt or "issue" in txt or "mag" in txt:
        return "magazine"
    elif "folder" in txt or "binder" in txt or "doc" in txt:
        return "folder"
    elif "book" in txt or "biography" in txt or "novel" in txt:
        return "book"
    elif "cabinet" in txt:
        return "cabinet"
    elif "bananas" == txt:
        return "bananas"
    elif "cupcake" == txt:
        return "cupcake"
    elif "cereal" == txt:
        return "cereal"
    elif "mincedmeat" == txt:
        return "mincedmeat"
    elif "apple" == txt:
        return "apple"
    elif "creamybuns" == txt:
        return "creamybuns"
    else:
        raise ValueError(f"Unknown query text: {txt}")
    
def _find_instance(query_text: str, query_cls: str, ref_image):
    """
    Find the instance UID of the object based on the query text.
    """
    global comm, vlm
    
    messages = []
    messages += [
        SystemMessage(content=(
           "You are a visual object-matching assistant. "
            "The user is looking for a specific object and will provide (1) a text description and (2) a reference image of the object as previously observed. "
            "Next, you will be shown several current camera views. Each view contains **red bounding boxes** labeled `Instance: {i}`. "
            "Your job is to determine whether any of the labeled instances match the reference object. "
            "If a match exists, reply with the **single most confident** instance ID (e.g., 0, 1, 2, ...). "
            "If no match is present, reply with **-1**. "
            "**Do not explain or justify your choice — reply with the integer only.**"
        ))
    ]
    messages += [
        HumanMessage(content=(
            f"The user is searching for: {query_text}. "
            "If any of the candidate instances match this object, reply with the matching instance ID. "
            "If none of them match, reply with -1. "
            "This is the reference image showing where the user last saw the object:"
        ))
    ]
    
    # Step 1: Encode reference image
    encoded_img = ros_image_to_base64(ref_image)
    ref_img_msg = [get_vlm_img_message(encoded_img)]
    
    messages += [HumanMessage(content=ref_img_msg)]
    
    # Step 2: Get images from simulator
    (ok_img, imgs) = _get_pano_images("normal")
    (ok_img, cls_imgs) = _get_pano_images("seg_class")
    (ok_img, inst_imgs) = _get_pano_images("seg_inst")

    # Step 3: Save for debug
    view_pil = display_grid_img(imgs + cls_imgs + inst_imgs, nrows=3)
    view_pil.save("../../outputs/debug_find_instance.png")

    # Step 4: Get scene graph
    success, graph = comm.environment_graph()
    
    # Step 5: Parse object color map 
    success, instance_colors = comm.instance_colors()
    
    # Step 6: Find IDs of all matching the target class objects
    target_ids = []
    for node in graph["nodes"]:
        if query_cls.lower() == node.get("class_name", "").lower():
            target_ids.append(str(node["id"]))
            
    # Step 7: Convert instance colors to uint8
    target_bgr_colors = []
    for uid in target_ids:
        rgb = instance_colors.get(uid)
        if rgb:
            bgr_uint8 = bgr_uint8 = (
                int(round(rgb[2] * 255)),  # B
                int(round(rgb[1] * 255)),  # G
                int(round(rgb[0] * 255))   # R
            ) 
            target_bgr_colors.append(bgr_uint8)

    print("Target instance IDs:", target_ids)
    print("Target colors:", target_bgr_colors)
    
    messages += [
        HumanMessage(content=(
            "Now, you will be shown several camera views: "
        ))
    ]
    
    # Step 8: Iterate over inst_imgs and draw boxes
    any_box_drawn = False  # <-- Add this
    valide_target_ids = []
    for i, (rgb_img, inst_img) in enumerate(zip(imgs, inst_imgs)):
        img_vis = rgb_img.copy()

        for uid, color in zip(target_ids, target_bgr_colors):
            mask = cv2.inRange(inst_img, np.array(color), np.array(color))  # exact match
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                any_box_drawn = True  # <-- Set if any contour found
                
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                # Draw bounding box
                cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 0, 255), 1)

                label = f"Instance ID: {uid}"
                valide_target_ids.append(uid)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1

                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                img_h, img_w = img_vis.shape[:2]

                # Try above the box
                above_y = y - 10
                if above_y - text_height >= 0:
                    text_y = above_y
                else:
                    # Otherwise, try below
                    below_y = y + h + text_height + 2
                    if below_y < img_h:
                        text_y = below_y
                    else:
                        # If both are out of bounds, clamp to bottom
                        text_y = max(0, min(y + h, img_h - text_height - 1))

                # Clamp x to stay fully within image width
                text_x = max(0, min(x, img_w - text_width - 1))

                cv2.putText(
                    img_vis,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 0, 255),
                    thickness,
                    cv2.LINE_AA
                )

        encoded_img = opencv_image_to_base64(img_vis)
        encoded_img = [get_vlm_img_message(encoded_img)]
        messages += [HumanMessage(content=encoded_img)]
        cv2.imwrite(f"../../outputs/seg_debug_view_{i}.png", img_vis)
        
    # Early return if no bounding boxes were drawn
    if not any_box_drawn:
        return None
    
    if len(valide_target_ids) == 1:
        return int(valide_target_ids[0])
        
    valid_instance_ids_str = ", ".join(valide_target_ids)
    messages += [
        HumanMessage(content=(
            f"Here are all valid instance IDs you may choose from: {valid_instance_ids_str}."
            " Please answer ONLY with one of these instance IDs, or -1 if there is no match."
        ))
    ]
    chat_prompt = ChatPromptTemplate.from_messages(messages)
    chained_model = chat_prompt | vlm
    
    instance_id = None
    for attempt in range(3):
        response = chained_model.invoke({})
        try:
            instance_id = int(response.content.strip())
            if instance_id in target_ids or instance_id == -1:
                break  # Success
        except Exception as e:
            if attempt == 3:
                raise ValueError(f"Invalid response from model: {response.content}") from e
    
    if instance_id == -1:
        instance_id = None
    
    return instance_id

def _get_visible_instances(class_list: dict) -> set[str]:  # NEW: pass class_list explicitly
    (ok_img, imgs)     = _get_pano_images("normal")
    (ok_img, cls_imgs) = _get_pano_images("seg_class")
    (ok_img, inst_imgs)= _get_pano_images("seg_inst")

    view_pil = display_grid_img(imgs + cls_imgs + inst_imgs, nrows=3)
    view_pil.save("../../outputs/debug_get_visible_instances.png")

    success, graph = comm.environment_graph()
    success, instance_colors = comm.instance_colors()
    assert len(imgs) == len(cls_imgs) == len(inst_imgs), "Number of images mismatch"

    # Precompute per-node colors for fast matching  --------------------------  # NEW
    id2node = {str(n["id"]): n for n in graph["nodes"]}
    id2inst_bgr = {}
    id2cls_bgr  = {}
    for uid, node in id2node.items():
        rgb_f = instance_colors.get(uid)
        if not rgb_f:
            continue
        inst_bgr = np.array([int(round(255*rgb_f[2])),
                             int(round(255*rgb_f[1])),
                             int(round(255*rgb_f[0]))], dtype=np.uint8)
        try:
            cls_bgr = np.array(semantic_cls_to_bgr(node["class_name"], class_list), dtype=np.uint8)
        except ValueError:
            continue
        id2inst_bgr[uid] = inst_bgr
        id2cls_bgr[uid]  = cls_bgr

    visible_instances: set[str] = set()
    MIN_PIX = 20

    # Frame loop --------------------------------------------------------------
    for img, cls_img, inst_img in zip(imgs, cls_imgs, inst_imgs):
        unique_inst_colors = np.unique(inst_img.reshape(-1, 3), axis=0)
        for inst_color in unique_inst_colors:
            if np.all(inst_color == 0):
                continue  # background

            # exact instance-color mask (fast & precise)
            mask_inst = cv2.inRange(inst_img, inst_color, inst_color)
            if cv2.countNonZero(mask_inst) < MIN_PIX:
                continue

            # majority class under this instance (sanity check)
            cls_pixels = cls_img[mask_inst.astype(bool)]
            if cls_pixels.size == 0:
                continue
            class_colors, counts = np.unique(cls_pixels.reshape(-1, 3), axis=0, return_counts=True)
            majority_cls = class_colors[np.argmax(counts)]

            # Find node with (exact) same instance color AND (exact) same class color  # NEW
            found = False
            for uid, inst_bgr in id2inst_bgr.items():
                if not np.array_equal(inst_color, inst_bgr):
                    continue
                if not np.array_equal(majority_cls, id2cls_bgr[uid]):
                    continue
                # Use prefab_name if that’s what you want to return; consider ID to avoid collisions
                visible_instances.add(id2node[uid].get("prefab_name", f"id:{uid}"))
                found = True
                break

            # (Optional) If you expect slight palette noise, switch to np.allclose(..., atol=2)

    return visible_instances
    
def handle_find_request(req):
    global comm
    rospy.loginfo("Received find request")
    
    find_success = False
    target_node_id = None
    target_position = None
    
    query_cls = _get_query_text(req.query_text.lower())
    if req.ref_image:
        target_node_id = _find_instance(req.query_text, query_cls, req.ref_image)
    else:
        target_node_id = find_target_node_id(query_cls)
        
    visible_instances = _get_visible_instances()
    
    success, graph = comm.environment_graph()
    
    if target_node_id is None:
        (ok_img, imgs) = _get_pano_images("normal")
        view_pil = display_grid_img(imgs, nrows=2)
        view_pil.save("../../outputs/debug_find.png")
        rospy.logwarn(f"Object '{query_cls}' not found in visible objects.")
        return FindObjectSrvResponse(success=False)
        
    find_success = target_node_id is not None
    target_node = extract_nodes_by_ids(graph["nodes"], [target_node_id])
    if target_node is None or len(target_node) == 0:
        return FindObjectSrvResponse(success=False)
    position = target_node[0]["obj_transform"]["position"]
    target_position = Point(position[0], position[1], position[2])
    
    return FindObjectSrvResponse(
        success=find_success,
        id=target_node_id,
        position=target_position,
        visible_instances=list(visible_instances),
    )
    
def handle_pick_request(req):
    global comm, pano_camera_select, first_person_pano_camera_select, tall_pano_camera_select
    
    target_node_id = None
    if req.instance_id is not None:
        _, graph = comm.environment_graph()
        target_node_id = int(req.instance_id)
        target_node = extract_nodes_by_ids(graph["nodes"], [target_node_id])
        if len(target_node) != 1:
            rospy.logwarn(f"Object not found in visible objects with instance ID {target_node_id}")
            return PickObjectSrvResponse(
                success=False,
            )
        target_node = target_node[0]
        query_text = target_node["class_name"]
        if target_node["class_name"].lower() != query_text.lower():
            rospy.logwarn(f"Object '{query_text}' does not match instance ID {target_node_id} class '{target_node['class_name']}'")
            return PickObjectSrvResponse(
                success=False,
            )
    else:
        query_text = _get_query_text(req.query_text.lower())
        target_node_id = find_target_node_id(query_text)
    
    if target_node_id is None:
        rospy.logwarn(f"Object '{query_text}' not found in visible objects.")
        return PickObjectSrvResponse(success=False)

    if _long_range_detect_enabled:
        if not _detect_instance(target_node_id, max_depth=PICK_OPEN_DEPTH_MAX):
            rospy.logwarn(
                f"Pick gated: object '{query_text}' (id={target_node_id}) is "
                f"farther than {PICK_OPEN_DEPTH_MAX}m or not visible."
            )
            return PickObjectSrvResponse(success=False)

    if pano_camera_select == first_person_pano_camera_select:
        script = [f"<char0> [Grab] <{query_text}> ({target_node_id})"]
        success, message = comm.render_script(script=script,
                                            processing_time_limit=60,
                                            find_solution=False,
                                            image_width=640,
                                            image_height=480,  
                                            skip_animation=True,
                                            recording=False,
                                            save_pose_data=False)
    else:
        success = _detect_instance(target_node_id)
    
    _, graph = comm.environment_graph()
    target_node = extract_nodes_by_ids(graph["nodes"], [target_node_id])[0]
    instance_uid = target_node["prefab_name"]
    
    return PickObjectSrvResponse(
        success=success,
        instance_uid=instance_uid
    )
    
def handle_open_request(req):
    global comm
    rospy.loginfo("Received open request")
    
    _, graph = comm.environment_graph()
    target_node_id = int(req.instance_id)
    target_node = extract_nodes_by_ids(graph["nodes"], [target_node_id])
    if len(target_node) < 1:
        rospy.logwarn(f"Object not found in visible objects with instance ID {target_node_id}")
        return OpenVirtualHomeObjectSrvResponse(
            success=False,
            instance_uid="",
            message=f"object with instance ID {target_node_id} not found in scene graph",
        )
    target_node = target_node[0]
    query_text = target_node["class_name"]

    if _long_range_detect_enabled:
        if not _detect_instance(target_node_id, max_depth=PICK_OPEN_DEPTH_MAX):
            rospy.logwarn(
                f"Open gated: object '{query_text}' (id={target_node_id}) is "
                f"farther than {PICK_OPEN_DEPTH_MAX}m or not visible."
            )
            return OpenVirtualHomeObjectSrvResponse(
                success=False,
                instance_uid=target_node.get("prefab_name", ""),
                message=f"object farther than {PICK_OPEN_DEPTH_MAX}m",
            )

    script = [f"<char0> [Open] <{query_text}> ({target_node_id})"]
    success, message = comm.render_script(script=script,
                                        processing_time_limit=60,
                                        find_solution=False,
                                        image_width=640,
                                        image_height=480,  
                                        skip_animation=True,
                                        recording=False,
                                        save_pose_data=False)

    # Opening a container/door changes scene state — any cached pano snapshot
    # is now stale. Force a fresh fetch on the next observe/detect.
    if success and _snapshot_obs_enabled:
        _snapshot_obs_invalidate()

    _, graph = comm.environment_graph()
    target_node = extract_nodes_by_ids(graph["nodes"], [target_node_id])[0]
    instance_uid = target_node["prefab_name"]

    return OpenVirtualHomeObjectSrvResponse(
        success=success,
        instance_uid=instance_uid,
        message=str(message)
    )

def _detect_instance(query_id: int, max_depth: float = None) -> bool:
    """
    Return True iff the specific instance (by node/uid) is visible in any pano view,
    AND its mean masked depth (ignoring zeros) is < DEPTH_MAX.

    If `max_depth` is None, DEPTH_MAX follows the --long_range_detect flag
    (5m when set, else 2m). Callers (pick/open gates) pass an explicit value
    to maintain their own depth cap regardless of the flag.
    """
    import numpy as np
    import cv2

    global comm, pano_camera_select

    # ── params
    MIN_PIX = 32
    MIN_W, MIN_H = 12, 12
    ATOL = 0            # palette tolerance for inst seg
    if max_depth is None:
        DEPTH_MAX = 5 if _long_range_detect_enabled else 2
    else:
        DEPTH_MAX = max_depth
    MIN_DEPTH_PIX = 20  # require at least this many valid (>0) depth pixels

    # 1) Fetch views
    ok_rgb,  rgb_imgs   = _get_pano_images("normal")
    ok_inst, inst_imgs  = _get_pano_images("seg_inst")
    ok_depth, depth_imgs= _get_pano_images("depth")
    if not (ok_rgb and ok_inst and ok_depth) or not rgb_imgs:
        return False

    # 2) Debug montage (best-effort)
    try:
        ok_cls, cls_imgs = _get_pano_images("seg_class")
        view_pil = display_grid_img(rgb_imgs + (cls_imgs if ok_cls else []) + inst_imgs, nrows=3 if ok_cls else 2)
        view_pil.save("../../outputs/debug_detect_instance.png")
    except Exception:
        pass

    # 3) Lookup instance color
    _, instance_colors = comm.instance_colors()
    uid = str(query_id)
    rgb_f = instance_colors.get(uid)  # float RGB [0,1]
    if not rgb_f:
        return False

    inst_bgr = np.array([int(round(255*rgb_f[2])),
                         int(round(255*rgb_f[1])),
                         int(round(255*rgb_f[0]))], dtype=np.uint8)

    for i, (rgb_img, inst_img, d) in enumerate(zip(rgb_imgs, inst_imgs, depth_imgs)):
        if inst_img is None or d is None:
            continue
        depth_scalar = d[..., 0]  # HxW

        # exact/tolerant instance mask
        if ATOL == 0:
            m_inst = cv2.inRange(inst_img, inst_bgr, inst_bgr)
        else:
            lo = np.clip(inst_bgr - ATOL, 0, 255).astype(np.uint8)
            hi = np.clip(inst_bgr + ATOL, 0, 255).astype(np.uint8)
            m_inst = cv2.inRange(inst_img, lo, hi)

        if cv2.countNonZero(m_inst) < MIN_PIX:
            continue

        cnts, _ = cv2.findContours(m_inst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < MIN_W or h < MIN_H:
                continue

            # depth mask for this contour
            obj_mask = np.zeros(m_inst.shape, dtype=np.uint8)
            cv2.drawContours(obj_mask, [c], -1, 255, thickness=cv2.FILLED)
            obj_depth = depth_scalar[obj_mask.astype(bool)]
            obj_depth = obj_depth[obj_depth > 0]  # ignore zeros
            if obj_depth.size < MIN_DEPTH_PIX:
                continue

            mean_depth = float(obj_depth.mean())
            if mean_depth < DEPTH_MAX:
                # Optional: annotate and save
                vis = rgb_img.copy()
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(vis, f"id:{uid} z~{mean_depth:.2f}m", (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imwrite(f"../../outputs/inst_depth_pass_view_{i}.png", vis)
                return True

    return False

def _detect_objects(query_cls: List[str]):
    """
    Detect all visible instances whose class_name ∈ query_cls.
    For each detected contour, compute mean depth on its mask (ignore zeros)
    and keep only those with mean depth < DEPTH_MAX.

    Returns: (set[str] of instance IDs, List[ROS Image] of RGB with boxes+depth)
    """
    import numpy as np
    import cv2
    import traceback
    import time
    from concurrent.futures import ThreadPoolExecutor

    global comm, pano_camera_select, class_list

    # ── params
    MIN_PIX = 32
    MIN_W, MIN_H = 12, 12
    ATOL = 0            # palette tolerance
    DEPTH_MAX = 5 if _long_range_detect_enabled else 2
    MIN_DEPTH_PIX = 20  # require some valid depth pixels

    # 1) Fetch views
    start_time = time.perf_counter()

    ok_rgb,  rgb_imgs   = _get_pano_images("normal")
    ok_cls,  cls_imgs   = _get_pano_images("seg_class")
    ok_inst, inst_imgs  = _get_pano_images("seg_inst")
    ok_depth, depth_imgs= _get_pano_images("depth")

    end_time = time.perf_counter()
    rospy.loginfo(f"Camera image fetch time: {(end_time - start_time):.2f} s")
    if not (ok_rgb and ok_cls and ok_inst and ok_depth) or not rgb_imgs:
        return (set(), [])

    # 2) Debug montage
    # try:
    #     view_pil = display_grid_img(rgb_imgs + cls_imgs + inst_imgs, nrows=3)
    #     view_pil.save("../../outputs/debug_detect_objects.png")
    # except Exception as e:
    #     rospy.logerr(f"Error in debug montage: {e}")
    #     traceback.print_exc()

    # 3) Scene graph & colors
    _, graph = comm.environment_graph()
    _, instance_colors = comm.instance_colors()

    want = {c.lower() for c in query_cls}
    id2node = {str(n["id"]): n for n in graph["nodes"]}

    target_ids: list[str] = []
    for n in graph["nodes"]:
        cname = (n.get("class_name") or "").lower()
        if cname in want:
            target_ids.append(str(n["id"]))
    if not target_ids:
        return (set(), [])

    def _cls_to_bgr(cname: str) -> np.ndarray:
        return np.array(semantic_cls_to_bgr(cname, class_list), dtype=np.uint8)

    id2_inst_bgr, id2_cls_bgr = {}, {}
    for uid in target_ids:
        rgb_f = instance_colors.get(uid)
        if not rgb_f:
            continue
        id2_inst_bgr[uid] = np.array([int(round(255*rgb_f[2])),
                                      int(round(255*rgb_f[1])),
                                      int(round(255*rgb_f[0]))], dtype=np.uint8)
        id2_cls_bgr[uid]  = _cls_to_bgr(id2node[uid]["class_name"])

    def _process_view(view_data):
        i, rgb_img, inst_img, cls_img, d = view_data
        vis = rgb_img.copy()

        if d is None or inst_img is None or cls_img is None:
            return set(), opencv_to_ros_image(vis)

        depth_scalar = d[..., 0] if (d.ndim == 3 and d.shape[2] >= 1) else d
        local_valid_target_ids: set[str] = set()

        for uid in list(id2_inst_bgr.keys()):
            inst_bgr = id2_inst_bgr[uid]
            cls_bgr = id2_cls_bgr[uid]

            if ATOL == 0:
                m_inst = cv2.inRange(inst_img, inst_bgr, inst_bgr)
                m_cls = cv2.inRange(cls_img, cls_bgr, cls_bgr)
            else:
                lo_i = np.clip(inst_bgr - ATOL, 0, 255).astype(np.uint8)
                hi_i = np.clip(inst_bgr + ATOL, 0, 255).astype(np.uint8)
                m_inst = cv2.inRange(inst_img, lo_i, hi_i)

                lo_c = np.clip(cls_bgr - ATOL, 0, 255).astype(np.uint8)
                hi_c = np.clip(cls_bgr + ATOL, 0, 255).astype(np.uint8)
                m_cls = cv2.inRange(cls_img, lo_c, hi_c)

            m = cv2.bitwise_and(m_inst, m_cls)
            if cv2.countNonZero(m) < MIN_PIX:
                continue

            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if w < MIN_W or h < MIN_H:
                    continue

                obj_mask = np.zeros(m.shape, dtype=np.uint8)
                cv2.drawContours(obj_mask, [c], -1, 255, thickness=cv2.FILLED)
                obj_depth = depth_scalar[obj_mask.astype(bool)]
                obj_depth = obj_depth[obj_depth > 0]
                if obj_depth.size < MIN_DEPTH_PIX:
                    continue

                mean_depth = float(obj_depth.mean())
                if mean_depth >= DEPTH_MAX:
                    continue

                local_valid_target_ids.add(uid)

                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
                label = f"ID:{uid} z~{mean_depth:.2f}m"
                font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1
                (tw, th_text), _ = cv2.getTextSize(label, font, fs, th)
                img_h, img_w = vis.shape[:2]
                ty = y - 10 if (y - 10 - th_text) >= 0 else min(y + h + th_text + 2, img_h - th_text - 1)
                tx = max(0, min(x, img_w - tw - 1))
                cv2.putText(vis, label, (tx, ty), font, fs, (0, 0, 255), th, cv2.LINE_AA)

        return local_valid_target_ids, opencv_to_ros_image(vis)

    valid_target_ids: set[str] = set()
    ros_images = []
    view_data = [(i, rgb_img, inst_img, cls_img, d)
                 for i, (rgb_img, inst_img, cls_img, d)
                 in enumerate(zip(rgb_imgs, inst_imgs, cls_imgs, depth_imgs))]

    if not view_data:
        return (set(), [])

    max_workers = min(6, len(view_data))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for local_ids, ros_img in pool.map(_process_view, view_data):
            valid_target_ids.update(local_ids)
            ros_images.append(ros_img)
        
    return (valid_target_ids, ros_images)


def handle_detect_virtualhome_request(req):
    global comm, class_list, pano_camera_select
    rospy.loginfo(f"Received detect virtual home object request: {req.query_text}")

    def _fetch_input_panos():
        # Fallback for failure paths: un-annotated RGB panos so the agent
        # still sees the current view rather than nothing.
        try:
            ok, rgb_imgs = _get_pano_images("normal")
            if not ok or not rgb_imgs:
                return []
            return [opencv_to_ros_image(img) for img in rgb_imgs]
        except Exception:
            return []

    try:
        query_cls = _get_query_text(req.query_text.lower())
        if query_cls == "cabinet":
            query_cls = ["kitchencabinet", "bathroomcabinet"]
        else:
            query_cls = [query_cls]
        instance_ids, ros_images = _detect_objects(query_cls)
        instance_ids = [int(id) for id in instance_ids]

        # visible_instances = _get_visible_instances(class_list)

        success = len(instance_ids) > 0
        if not ros_images:
            ros_images = _fetch_input_panos()
        return DetectVirtualHomeObjectSrvResponse(
            success=success,
            ids=instance_ids,
            # visible_instances=list(visible_instances),
            images=ros_images
        )
    except Exception as e:
        rospy.logerr(f"Error in detect_virtual_home_object request: {e}")
        import traceback; traceback.print_exc()
        return DetectVirtualHomeObjectSrvResponse(
            success=False,
            images=_fetch_input_panos(),
        )


CHANGE_SCENE_MAX_ATTEMPTS = 2
CHANGE_SCENE_RETRY_BACKOFF_S = 5.0


def handle_virtualhome_scene_request(req):
    global comm, cameras_select, pano_camera_select, tall_pano_camera_select, first_person_pano_camera_select
    rospy.loginfo(f"Received change virtual home graph request: {req.graph_path}")

    with open(req.graph_path, "r") as f:
        graph = json.load(f)

    if graph is None:
        rospy.logerr(
            f"change_virtualhome_graph: graph JSON at {req.graph_path} parsed to None"
        )
        return ChangeVirtualHomeGraphSrvResponse(success=False)

    for attempt in range(1, CHANGE_SCENE_MAX_ATTEMPTS + 1):
        try:
            if req.scene_id is not None:
                comm.reset(req.scene_id)
            else:
                comm.reset()
            success, message = comm.expand_scene(graph)
            if not success:
                rospy.logerr(
                    f"change_virtualhome_graph (scene_id={req.scene_id}): "
                    f"comm.expand_scene failed: {message}"
                )
                return ChangeVirtualHomeGraphSrvResponse(success=False)

            s, nc_before = comm.camera_count()
            prepare_pano_character_camera(comm)
            prepare_tall_pano_character_camera(comm)
            comm.add_character('chars/Female2', initial_room='bathroom')
            s, nc_after = comm.camera_count()
            cameras_select = list(range(nc_before, nc_after))
            pano_camera_select = cameras_select[8:14]
            first_person_pano_camera_select = cameras_select[8:14]
            tall_pano_camera_select = cameras_select[14:20]

            rospy.loginfo(
                f"VirtualHome scene updated (scene_id={req.scene_id}). "
            )
            if _snapshot_obs_enabled:
                _snapshot_obs_invalidate()
                _snapshot_obs_populate()
            return ChangeVirtualHomeGraphSrvResponse(success=success)
        except UnityEngineException as e:
            status_code = e.args[0] if e.args else None
            if status_code != 408:
                rospy.logwarn(
                    f"change_virtualhome_graph (scene_id={req.scene_id}) failed with non-408 "
                    f"UnityEngineException — not retrying: {e.message}"
                )
                return ChangeVirtualHomeGraphSrvResponse(success=False)
            if attempt >= CHANGE_SCENE_MAX_ATTEMPTS:
                rospy.logwarn(
                    f"change_virtualhome_graph (scene_id={req.scene_id}) hit Unity 408 on "
                    f"attempt {attempt}/{CHANGE_SCENE_MAX_ATTEMPTS} — giving up: {e.message}"
                )
                return ChangeVirtualHomeGraphSrvResponse(success=False)
            rospy.logwarn(
                f"change_virtualhome_graph (scene_id={req.scene_id}) hit Unity 408 on "
                f"attempt {attempt}/{CHANGE_SCENE_MAX_ATTEMPTS} — sleeping "
                f"{CHANGE_SCENE_RETRY_BACKOFF_S:.1f}s and retrying."
            )
            time.sleep(CHANGE_SCENE_RETRY_BACKOFF_S)
    return ChangeVirtualHomeGraphSrvResponse(success=False)

if __name__ == "__main__":
    args = parse_args()
    if args.parallel:
        rospy.init_node(f'virtualhome_ros_{args.port}', anonymous=True)
    else:
        rospy.init_node('virtualhome_ros', anonymous=True)

    _snapshot_obs_enabled = bool(args.snapshot_obs)
    if _snapshot_obs_enabled:
        rospy.loginfo("snapshot_obs: ENABLED (pano cache active for observe/detect/find/pick)")

    _long_range_detect_enabled = bool(args.long_range_detect)
    if _long_range_detect_enabled:
        rospy.loginfo(
            f"long_range_detect: ENABLED (detection DEPTH_MAX=5m; "
            f"pick/open gated at {PICK_OPEN_DEPTH_MAX}m)"
        )

    _verbose_enabled = bool(args.verbose)
    if _verbose_enabled:
        rospy.loginfo("verbose: ENABLED (observe() saves debug pano grid to ../../outputs/debug_observe.png)")

    prefab_classes, class_list = load_prefab_metadata("../resources/PrefabClass.json")

    comm = UnityCommunication(port=args.port)
    # A healthy pano render is seconds; a wedged Unity renderer never returns.
    # 150s is long enough to ride out a contended-GPU slow render but short
    # enough that a true wedge is detected (and the sim restarted) in ~2.5min
    # instead of 5. See start_sims.py watchdog.
    comm.timeout_wait = 150

    navigate_service = get_moma_service_name(args.port, 'navigate', args.parallel)
    observe_service = get_moma_service_name(args.port, 'observe', args.parallel)
    visible_objects_service = get_moma_service_name(args.port, 'visible_objects', args.parallel)
    find_object_service = get_moma_service_name(args.port, 'find_object', args.parallel)
    pick_object_service = get_moma_service_name(args.port, 'pick_object', args.parallel)
    open_object_service = get_moma_service_name(args.port, 'open_object', args.parallel)
    detect_virtual_home_object_service = get_moma_service_name(args.port, 'detect_virtual_home_object', args.parallel)
    change_virtualhome_graph_service = get_moma_service_name(args.port, 'change_virtualhome_graph', args.parallel)
    
    rospy.Service(navigate_service, GetImageAtPoseSrv, handle_navigate_request)
    rospy.loginfo(f"Ready to navigate: {navigate_service}")
    rospy.Service(observe_service, GetImageSrv, handle_observe_request)
    rospy.loginfo(f"Ready to observe: {observe_service}")
    rospy.Service(visible_objects_service, GetVisibleObjectsSrv, handle_visible_objects_request)
    rospy.loginfo(f"Ready to return visible objects: {visible_objects_service}")
    rospy.Service(find_object_service, FindObjectSrv, handle_find_request)
    rospy.loginfo(f"Ready to find objects: {find_object_service}")
    rospy.Service(pick_object_service, PickObjectSrv, handle_pick_request)
    rospy.loginfo(f"Ready to pick objects: {pick_object_service}")
    rospy.Service(open_object_service, OpenVirtualHomeObjectSrv, handle_open_request)
    rospy.loginfo(f"Ready to open virtual home objects: {open_object_service}")
    rospy.Service(detect_virtual_home_object_service, DetectVirtualHomeObjectSrv, handle_detect_virtualhome_request)
    rospy.loginfo(f"Ready to detect virtual home objects: {detect_virtual_home_object_service}")
    rospy.Service(change_virtualhome_graph_service, ChangeVirtualHomeGraphSrv, handle_virtualhome_scene_request)
    rospy.loginfo(f"Ready to change virtual home graph: {change_virtualhome_graph_service}")
    
    rospy.spin()