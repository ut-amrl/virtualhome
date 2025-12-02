import argparse
import json
import sys
from PIL import ImageDraw
import copy
import numpy as np
import cv2

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Simulation
sys.path.append('../simulation')
from unity_simulator.comm_unity import UnityCommunication
from unity_simulator import utils_viz
from ros_utils import *
from utils_demo import *
from graph_utils import *

## ROS Service Calls
import rospy
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.srv import (
    GetImageSrv,
    GetImageSrvResponse,
    GetImageAtPoseSrv, 
    GetImageAtPoseSrvResponse, 
    PickObjectSrv, 
    PickObjectSrvResponse,
    GetVisibleObjectsSrv,
    GetVisibleObjectsSrvResponse,
    FindObjectSrv,
    FindObjectSrvResponse,
    SemanticObjectDetectionSrv,
    SemanticObjectDetectionSrvRequest,
    SemanticObjectDetectionSrvResponse,
    ChangeVirtualHomeGraphSrv,
    ChangeVirtualHomeGraphSrvResponse,
    DetectVirtualHomeObjectSrv,
    DetectVirtualHomeObjectSrvRequest,
    DetectVirtualHomeObjectSrvResponse,
    OpenVirtualHomeObjectSrv,
    OpenVirtualHomeObjectSrvRequest,
    OpenVirtualHomeObjectSrvResponse,
)
from geometry_msgs.msg import Point

comm = None
class_list = None
cameras_select = None
pano_camera_select = None
first_person_pano_camera_select = None
tall_pano_camera_select = None
vlm = None

def parse_args():
    parser = argparse.ArgumentParser(description='Virtual Home ROS Service')
    parser.add_argument('--port', type=str, required=True, help='Port for Unity communication')
    # parser.add_argument("--graph_path", type=str, required=True, help="Path to the scene graph")
    return parser.parse_args()

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
    
    (ok_img, imgs) = comm.camera_image(pano_camera_select, mode="normal")
    if ok_img:
        view_pil = display_grid_img(imgs, nrows=2)
        view_pil.save("../../outputs/debug_observe.png")
    
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
        if not success:
            return GetImageAtPoseSrvResponse(success=False)
        if z > 0.3:
            pano_camera_select = copy.deepcopy(tall_pano_camera_select)
        else:
            pano_camera_select = copy.deepcopy(first_person_pano_camera_select)
        pano_images = observe()
        return GetImageAtPoseSrvResponse(success=success, pano_images=pano_images)
    except:
        import pdb; pdb.set_trace()

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
    return txt.lower().strip()
    
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
    (ok_img, imgs) = comm.camera_image(pano_camera_select, mode="normal")
    (ok_img, cls_imgs) = comm.camera_image(pano_camera_select, mode="seg_class")
    (ok_img, inst_imgs) = comm.camera_image(pano_camera_select, mode="seg_inst")

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
    (ok_img, imgs)     = comm.camera_image(pano_camera_select, mode="normal")
    (ok_img, cls_imgs) = comm.camera_image(pano_camera_select, mode="seg_class")
    (ok_img, inst_imgs)= comm.camera_image(pano_camera_select, mode="seg_inst")

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
        (ok_img, imgs) = comm.camera_image(pano_camera_select, mode="normal")
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
        return PickObjectSrvResponse(
            success=False,
        )
    target_node = target_node[0]
    query_text = target_node["class_name"]
    
    script = [f"<char0> [Open] <{query_text}> ({target_node_id})"]
    success, message = comm.render_script(script=script,
                                        processing_time_limit=60,
                                        find_solution=False,
                                        image_width=640,
                                        image_height=480,  
                                        skip_animation=True,
                                        recording=False,
                                        save_pose_data=False)
    
    _, graph = comm.environment_graph()
    target_node = extract_nodes_by_ids(graph["nodes"], [target_node_id])[0]
    instance_uid = target_node["prefab_name"]
    
    return OpenVirtualHomeObjectSrvResponse(
        success=success,
        instance_uid=instance_uid,
        message=str(message)
    )

def _detect_instance(query_id: int) -> bool:
    """
    Return True iff the specific instance (by node/uid) is visible in any pano view,
    AND its mean masked depth (ignoring zeros) is < DEPTH_MAX.
    """
    import numpy as np
    import cv2

    global comm, pano_camera_select

    # ── params
    MIN_PIX = 32
    MIN_W, MIN_H = 12, 12
    ATOL = 0            # palette tolerance for inst seg
    DEPTH_MAX = 2.5     # meters (cap)
    MIN_DEPTH_PIX = 20  # require at least this many valid (>0) depth pixels

    # 1) Fetch views
    ok_rgb,  rgb_imgs   = comm.camera_image(pano_camera_select, mode="normal")
    ok_inst, inst_imgs  = comm.camera_image(pano_camera_select, mode="seg_inst")
    ok_depth, depth_imgs= comm.camera_image(pano_camera_select, mode="depth")
    if not (ok_rgb and ok_inst and ok_depth) or not rgb_imgs:
        return False

    # 2) Debug montage (best-effort)
    try:
        ok_cls, cls_imgs = comm.camera_image(pano_camera_select, mode="seg_class")
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

    global comm, pano_camera_select, class_list

    # ── params
    MIN_PIX = 32
    MIN_W, MIN_H = 12, 12
    ATOL = 0            # palette tolerance
    DEPTH_MAX = 2.5     # meters
    MIN_DEPTH_PIX = 20  # require some valid depth pixels

    # 1) Fetch views
    ok_rgb,  rgb_imgs   = comm.camera_image(pano_camera_select, mode="normal")
    ok_cls,  cls_imgs   = comm.camera_image(pano_camera_select, mode="seg_class")
    ok_inst, inst_imgs  = comm.camera_image(pano_camera_select, mode="seg_inst")
    ok_depth, depth_imgs= comm.camera_image(pano_camera_select, mode="depth")
    if not (ok_rgb and ok_cls and ok_inst and ok_depth) or not rgb_imgs:
        return (set(), [])

    # 2) Debug montage
    try:
        view_pil = display_grid_img(rgb_imgs + cls_imgs + inst_imgs, nrows=3)
        view_pil.save("../../outputs/debug_detect_objects.png")
    except Exception:
        pass

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

    valid_target_ids: set[str] = set()
    ros_images = []

    # 4) Per-view processing
    for i, (rgb_img, inst_img, cls_img, d) in enumerate(zip(rgb_imgs, inst_imgs, cls_imgs, depth_imgs)):
        vis = rgb_img.copy()
        depth_scalar = d[..., 0]  # HxW

        for uid in list(id2_inst_bgr.keys()):
            inst_bgr = id2_inst_bgr[uid]
            cls_bgr  = id2_cls_bgr[uid]

            # build masks (inst & class)
            if ATOL == 0:
                m_inst = cv2.inRange(inst_img, inst_bgr, inst_bgr)
                m_cls  = cv2.inRange(cls_img,  cls_bgr,  cls_bgr)
            else:
                lo_i = np.clip(inst_bgr - ATOL, 0, 255).astype(np.uint8)
                hi_i = np.clip(inst_bgr + ATOL, 0, 255).astype(np.uint8)
                m_inst = cv2.inRange(inst_img, lo_i, hi_i)

                lo_c = np.clip(cls_bgr - ATOL, 0, 255).astype(np.uint8)
                hi_c = np.clip(cls_bgr + ATOL, 0, 255).astype(np.uint8)
                m_cls  = cv2.inRange(cls_img,  lo_c, hi_c)

            m = cv2.bitwise_and(m_inst, m_cls)
            if cv2.countNonZero(m) < MIN_PIX:
                continue

            # contours for this uid
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if w < MIN_W or h < MIN_H:
                    continue

                # depth for this contour
                obj_mask = np.zeros(m.shape, dtype=np.uint8)
                cv2.drawContours(obj_mask, [c], -1, 255, thickness=cv2.FILLED)
                obj_depth = depth_scalar[obj_mask.astype(bool)]
                obj_depth = obj_depth[obj_depth > 0]  # ignore zeros
                if obj_depth.size < MIN_DEPTH_PIX:
                    continue

                mean_depth = float(obj_depth.mean())
                if mean_depth >= DEPTH_MAX:
                    continue  # cap by depth

                valid_target_ids.add(uid)

                # annotate
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
                label = f"ID:{uid} z~{mean_depth:.2f}m"
                font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                (tw, th_text), _ = cv2.getTextSize(label, font, fs, th)
                img_h, img_w = vis.shape[:2]
                ty = y - 10 if (y - 10 - th_text) >= 0 else min(y + h + th_text + 2, img_h - th_text - 1)
                tx = max(0, min(x, img_w - tw - 1))
                cv2.putText(vis, label, (tx, ty), font, fs, (0, 0, 255), th, cv2.LINE_AA)

        cv2.imwrite(f"../../outputs/seg_debug_view_{i}.png", vis)
        ros_images.append(opencv_to_ros_image(vis))
        
    return (valid_target_ids, ros_images)


def handle_detect_virtualhome_request(req):
    global comm, class_list
    rospy.loginfo(f"Received detect virtual home object request: {req.query_text}")

    try:
        query_cls = _get_query_text(req.query_text.lower())
        if query_cls == "cabinet":
            query_cls = ["kitchencabinet", "bathroomcabinet"]
        else:
            query_cls = [query_cls]
        instance_ids, ros_images = _detect_objects(query_cls)
        instance_ids = [int(id) for id in instance_ids]
        
        visible_instances = _get_visible_instances(class_list)
        
        return DetectVirtualHomeObjectSrvResponse(
            success=len(instance_ids) > 0,
            ids=instance_ids,
            visible_instances=list(visible_instances),
            images=ros_images
        )
    except Exception as e:
        return DetectVirtualHomeObjectSrvResponse(success=False)
    
def handle_virtualhome_scene_request(req):
    global comm, cameras_select, pano_camera_select, tall_pano_camera_select, first_person_pano_camera_select
    rospy.loginfo(f"Received change virtual home graph request: {req.graph_path}")
    
    with open(req.graph_path, "r") as f:
        graph = json.load(f)
    
    if graph is None:
        import pdb; pdb.set_trace()
        return ChangeVirtualHomeGraphSrvResponse(success=False)
    
    if req.scene_id is not None:
        comm.reset(req.scene_id)
    else:
        comm.reset()
    success, message = comm.expand_scene(graph)
    if not success:
        import pdb; pdb.set_trace()
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
    
    return ChangeVirtualHomeGraphSrvResponse(success=success)

if __name__ == "__main__":
    rospy.init_node('virtualhome_ros', anonymous=True)
    
    os.makedirs("../../outputs", exist_ok=True)
    
    args = parse_args()
    prefab_classes, class_list = load_prefab_metadata("../resources/PrefabClass.json")
    
    comm = UnityCommunication(port=args.port)
    comm.timeout_wait = 300
    
    vlm = ChatOpenAI(model="o3", temperature=1, api_key=os.environ.get("OPENAI_API_KEY"))
        
    rospy.Service('/moma/navigate', GetImageAtPoseSrv, handle_navigate_request)
    rospy.loginfo("Ready to navigate")
    rospy.Service('/moma/observe', GetImageSrv, handle_observe_request)
    rospy.loginfo("Ready to observe")
    rospy.Service('/moma/visible_objects', GetVisibleObjectsSrv, handle_visible_objects_request)
    rospy.loginfo("Ready to return visible objects")
    rospy.Service('/moma/find_object', FindObjectSrv, handle_find_request)
    rospy.loginfo("Ready to find objects")
    rospy.Service('/moma/pick_object', PickObjectSrv, handle_pick_request)
    rospy.loginfo("Ready to pick objects")
    rospy.Service('/moma/open_object', OpenVirtualHomeObjectSrv, handle_open_request)
    rospy.loginfo("Ready to open virtual home objects")
    rospy.Service('/moma/detect_virtual_home_object', DetectVirtualHomeObjectSrv, handle_detect_virtualhome_request)
    rospy.loginfo("Ready to detect virtual home objects")
    rospy.Service('/moma/change_virtualhome_graph', ChangeVirtualHomeGraphSrv, handle_virtualhome_scene_request)
    rospy.loginfo("Ready to change virtual home graph")
    
    rospy.spin()