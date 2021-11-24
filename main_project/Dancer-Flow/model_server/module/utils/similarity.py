from sklearn.metrics.pairwise import cosine_similarity

def single_score_similarity(user_pose_result, target_pose_result, user_frame=0,  target_frame=0):
    """
    input.
        user_pose_result : Pose data of the video uploaded by the user.
        target_pose_result : Video selected by the user for play.
        user_frame : default = 0frame.
        target_frame : default = 0frame.
    
    output
        score{
            face_body_score : int()
            left_arm_score : int()
            right_arm_score : int()
            left_leg_score : int()
            right_leg_score : int()
        }
    """
    upd = user_pose_result
    tpd = target_pose_result

    kupd = upd["keypoints"]
    ktpd = tpd["keypoints"]

    bx1_1 = upd["bbox"][0]
    by1_1 = upd["bbox"][1]
    bx1_2 = upd["bbox"][2]
    by1_2 = upd["bbox"][3]

    bx2_1 = tpd["bbox"][0]
    by2_1 = tpd["bbox"][1]
    bx2_2 = tpd["bbox"][2]
    by2_2 = tpd["bbox"][3]
    
    center1 = [(bx1_1 + bx1_2) / 2, (by1_1 + by1_2) / 2]
    center2 = [(bx2_1 + bx2_2) / 2, (by2_1 + by2_2) / 2]
    
    

    #User_pose_result
    #얼굴+몸
    nose_key1                   = kupd["nose"]
    left_eye_key1               = kupd["left_eye"]
    right_eye_key1              = kupd["right_eye"]
    left_ear_key1               = kupd["left_ear"]
    right_ear_key1              = kupd["right_ear"]
    left_shoulder_key1          = kupd["left_shoulder"]
    right_shoulder_key1         = kupd["right_shoulder"]
    left_hip_key1               = kupd["left_hip"]
    right_hip_key1              = kupd["right_hip"]
    face_body_key_x1            = (nose_key1[0] + left_eye_key1[0] + right_eye_key1[0] 
                                + left_ear_key1[0] + right_ear_key1[0] + left_shoulder_key1[0]
                                + right_shoulder_key1[0]+ left_hip_key1[0]+ right_hip_key1[0])/9
    face_body_key_y1            = (nose_key1[1] + left_eye_key1[1] + right_eye_key1[1]
                                +left_ear_key1[1] + right_ear_key1[1] + left_shoulder_key1[1]
                                + right_shoulder_key1[1] + left_hip_key1[1] + right_hip_key1[1])/9
    face_body_key_nm_x1         = (face_body_key_x1 - center1[0]) / (bx1_2 - bx1_1)
    face_body_key_nm_y1         = (face_body_key_y1 - center1[1]) / (by1_2 - by1_1)
    
    #왼팔
    left_elbow_key1             = kupd["left_elbow"]
    left_wrist1                 = kupd["left_wrist"]
    left_arm_key_x1             = ((left_elbow_key1[0]+ left_wrist1[0])/2)
    left_arm_key_y1             = ((left_elbow_key1[1]+ left_wrist1[1])/2)
    left_arm_key_nm_x1          = (left_arm_key_x1-center1[0])/(bx1_2 - bx1_1)
    left_arm_key_nm_y1          = (left_arm_key_y1-center1[1])/(by1_2-by1_1)
    #오른팔
    right_elbow_key1            = kupd["right_elbow"]
    right_wrist1                = kupd["right_wrist"]
    right_arm_key_x1            = ((right_elbow_key1[0]+ right_wrist1[0])/2)
    right_arm_key_y1            = ((right_elbow_key1[1]+ right_wrist1[1])/2)
    right_arm_key_nm_x1         = (right_arm_key_x1-center1[0])/(bx1_2 - bx1_1)
    right_arm_key_nm_y1         = (right_arm_key_y1-center1[1])/(by1_2-by1_1)    
    #왼다리
    left_knee_key1              = kupd["left_knee"]
    left_ankle_key1             = kupd["left_ankle"]
    left_leg_key_x1             = ((left_knee_key1[0]+left_ankle_key1[0])/2)
    left_leg_key_y1             = ((left_knee_key1[1]+left_ankle_key1[1])/2)
    left_leg_key_nm_x1          = (left_leg_key_x1-center1[0])/(bx1_2 - bx1_1)
    left_leg_key_nm_y1          = (left_leg_key_y1-center1[1])/(by1_2-by1_1)
    #오른다리
    right_knee_key1             = kupd["right_knee"]
    right_ankle_key1            = kupd["right_ankle"]
    right_leg_key_x1            = ((right_knee_key1[0]+right_ankle_key1[0])/2)
    right_leg_key_y1            = ((right_knee_key1[1]+right_ankle_key1[1])/2)
    right_leg_key_nm_x1         = (right_leg_key_x1-center1[0])/(bx1_2 - bx1_1)
    right_leg_key_nm_y1         = (right_leg_key_y1-center1[1])/(by1_2 - by1_1)

    #target_pose_result
    #얼굴+몸
    nose_key2                   = ktpd["nose"]
    left_eye_key2               = ktpd["left_eye"]
    right_eye_key2              = ktpd["right_eye"]
    left_ear_key2               = ktpd["left_ear"]
    right_ear_key2              = ktpd["right_ear"]
    left_shoulder_key2          = ktpd["left_shoulder"]
    right_shoulder_key2         = ktpd["right_shoulder"]
    left_hip_key2               = ktpd["left_hip"]
    right_hip_key2              = ktpd["right_hip"] 
    face_body_key_x2            = (nose_key2[0] + left_eye_key2[0] + right_eye_key2[0] 
                                + left_ear_key2[0] + right_ear_key2[0] + left_shoulder_key2[0]
                                +right_shoulder_key2[0] + left_hip_key2[0] + right_hip_key2[0])/9
    face_body_key_y2            = (nose_key2[1] + left_eye_key2[1] + right_eye_key2[1] 
                                + left_ear_key2[1] + right_ear_key2[1] + left_shoulder_key2[1]
                                + right_shoulder_key2[1] + left_hip_key2[1] + right_hip_key2[1])/9
    face_body_key_nm_x2         = (face_body_key_x2 - center2[0]) / (bx2_2 - bx2_1)
    face_body_key_nm_y2         = (face_body_key_y2 - center2[1]) / (by2_2 - by2_1)
    #왼팔
    left_elbow_key2             = ktpd["left_elbow"]
    left_wrist2                 = ktpd["left_wrist"]
    left_arm_key_x2             = ((left_elbow_key2[0]+ left_wrist2[0])/2)
    left_arm_key_y2             = ((left_elbow_key2[1]+ left_wrist2[1])/2)
    left_arm_key_nm_x2          = (left_arm_key_x2-center2[0])/(bx2_2 - bx2_1)
    left_arm_key_nm_y2          = (left_arm_key_y2-center2[1])/(by2_2-by2_1)
    #오른팔
    right_elbow_key2            = ktpd["right_elbow"]
    right_wrist2                = ktpd["right_wrist"]
    right_arm_key_x2            = ((right_elbow_key2[0]+ right_wrist2[0])/2)
    right_arm_key_y2            = ((right_elbow_key2[1]+ right_wrist2[1])/2)
    right_arm_key_nm_x2         = (right_arm_key_x2-center2[0])/(bx2_2 - bx2_1)
    right_arm_key_nm_y2         = (right_arm_key_y2-center2[1])/(by2_2-by2_1)
    #왼다리
    left_knee_key2              = ktpd["left_knee"]
    left_ankle_key2             = ktpd["left_ankle"]
    left_leg_key_x2             = ((left_knee_key2[0]+left_ankle_key2[0])/2)
    left_leg_key_y2             = ((left_knee_key2[1]+left_ankle_key2[1])/2)
    left_leg_key_nm_x2          = (left_leg_key_x2-center2[0])/(bx2_2 - bx2_1)
    left_leg_key_nm_y2          = (left_leg_key_y2-center2[1])/(by2_2-by2_1)
    #오른다리
    right_knee_key2             = ktpd["right_knee"]
    right_ankle_key2            = ktpd["right_ankle"]
    right_leg_key_x2            = ((right_knee_key2[0]+right_ankle_key2[0])/2)
    right_leg_key_y2            = ((right_knee_key2[1]+right_ankle_key2[1])/2)
    right_leg_key_nm_x2         = (right_leg_key_x2-center2[0])/(bx2_2 - bx2_1)
    right_leg_key_nm_y2         = (right_leg_key_y2-center2[1])/(by2_2-by2_1)
    
    face_body_score     = cosine_similarity([[face_body_key_nm_x1, face_body_key_nm_y1]],[[face_body_key_nm_x2, face_body_key_nm_y2]])  
    left_arm_score      = cosine_similarity([[left_arm_key_nm_x1, left_arm_key_nm_y1]],[[left_arm_key_nm_x2, left_arm_key_nm_y2]])
    right_arm_score     = cosine_similarity([[right_arm_key_nm_x1, right_arm_key_nm_y1]],[[right_arm_key_nm_x2, right_arm_key_nm_y2]])
    left_leg_score      = cosine_similarity([[left_leg_key_nm_x1, left_leg_key_nm_y1]],[[left_leg_key_nm_x2, left_leg_key_nm_y2]])
    right_leg_score     = cosine_similarity([[right_leg_key_nm_x1, right_leg_key_nm_y1]],[[right_leg_key_nm_x2, right_leg_key_nm_y2]])
    
    result = dict()
    result["face_body_score"]   = face_body_score
    result["left_arm_score"]    = left_arm_score
    result["right_arm_score"]   = right_arm_score
    result["left_leg_score"]    = left_leg_score
    result["right_leg_score"]   = right_leg_score
    return result

def multi_score_similarity():
    pass

if __name__== "__Main__":
    single_score_similarity()
    pass