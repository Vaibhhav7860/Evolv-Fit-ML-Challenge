# Importing Necessary Modules

import cv2
import numpy as np
import face_recognition


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Loading a sample picture of Bhuvneshwar Kumar and learning how to recognize it.
bhuvneshwar_kumar = face_recognition.load_image_file("bhuvneshwar_kumar.jpg")
bhuvneshwar_kumar_face_encoding = face_recognition.face_encodings(bhuvneshwar_kumar)[0]

# Loading a sample picture of Dinesh Karthik and learning how to recognize it.
dinesh_karthik = face_recognition.load_image_file("dinesh_karthik.jpg")
dinesh_karthik_face_encoding = face_recognition.face_encodings(dinesh_karthik)[0]

# Loading a sample picture of Hardik Pandya and learning how to recognize it.
hardik_pandya = face_recognition.load_image_file("hardik_pandya.jpg")
hardik_pandya_face_encoding = face_recognition.face_encodings(hardik_pandya)[0]

# Loading a sample picture of Jasprit Bumrah and learning how to recognize it.
jasprit_bumrah = face_recognition.load_image_file("jasprit_bumrah.jpg")
jasprit_bumrah_face_encoding = face_recognition.face_encodings(jasprit_bumrah)[0]

# Loading a sample picture of K.L. Rahul and learning how to recognize it.
k_l_rahul = face_recognition.load_image_file("k_l_rahul.jpg")
k_l_rahul_face_encoding = face_recognition.face_encodings(k_l_rahul)[0]

# Loading a sample picture of Kedar Jadhav and learning how to recognize it.
kedar_jadhav = face_recognition.load_image_file("kedar_jadhav.jpg")
kedar_jadhav_face_encoding = face_recognition.face_encodings(kedar_jadhav)[0]

# Loading a sample picture of Kuldeep Yadav and learning how to recognize it.
kuldeep_yadav = face_recognition.load_image_file("kuldeep_yadav.jpg")
kuldeep_yadav_face_encoding = face_recognition.face_encodings(kuldeep_yadav)[0]

# Loading a sample picture of Mohammed Shami and learning how to recognize it.
mohammed_shami = face_recognition.load_image_file("mohammed_shami.jpg")
mohammed_shami_face_encoding = face_recognition.face_encodings(mohammed_shami)[0]

# Loading a sample picture of MS Dhoni and learning how to recognize it.
ms_dhoni = face_recognition.load_image_file("ms_dhoni.jpg")
ms_dhoni_face_encoding = face_recognition.face_encodings(ms_dhoni)[0]

# Load a sample picture of Ravindra Jadeja and learning how to recognize it.
ravindra_jadeja = face_recognition.load_image_file("ravindra_jadeja.jpg")
ravindra_jadeja_face_encoding = face_recognition.face_encodings(ravindra_jadeja)[0]

# Loading a sample picture of Rohit Sharma and learning how to recognize it.
rohit_sharma = face_recognition.load_image_file("rohit_sharma.jpg")
rohit_sharma_face_encoding = face_recognition.face_encodings(rohit_sharma)[0]

# Loading a sample picture of Shikhar Dhawan and learning how to recognize it.
shikhar_dhawan = face_recognition.load_image_file("shikhar_dhawan.jpg")
shikhar_dhawan_face_encoding = face_recognition.face_encodings(shikhar_dhawan)[0]

# Loading a sample picture of Vijay Shankar and learning how to recognize it.
vijay_shankar = face_recognition.load_image_file("vijay_shankar.jpg")
vijay_shankar_face_encoding = face_recognition.face_encodings(vijay_shankar)[0]

# Loading a sample picture of Virat Kohli and learning how to recognize it.
virat_kohli = face_recognition.load_image_file("virat_kohli.jpg")
virat_kohli_face_encoding = face_recognition.face_encodings(virat_kohli)[0]

# Loading a sample picture of Yuzvendra Chahal and learn how to recognize it.
yuzvendra_chahal = face_recognition.load_image_file("yuzvendra_chahal.jpg")
yuzvendra_chahal_face_encoding = face_recognition.face_encodings(yuzvendra_chahal)[0]

# Creating a list of known face encodings of players with their names.
known_face_encodings = [
    bhuvneshwar_kumar_face_encoding,
    dinesh_karthik_face_encoding,
    hardik_pandya_face_encoding,
    jasprit_bumrah_face_encoding,
    k_l_rahul_face_encoding,
    kedar_jadhav_face_encoding,
    kuldeep_yadav_face_encoding,
    mohammed_shami_face_encoding,
    ms_dhoni_face_encoding,
    ravindra_jadeja_face_encoding,
    rohit_sharma_face_encoding,
    shikhar_dhawan_face_encoding,
    vijay_shankar_face_encoding,
    virat_kohli_face_encoding,
    yuzvendra_chahal_face_encoding
]
# Creating a list of known face names of players which will act as labels.
known_face_names = [
    "Bhuvneshwar Kumar",
    "Dinesh Karthik",
    "Hardik Pandya",
    "Jasprit Bumrah",
    "K.L. Rahul",
    "Kedar Jadhav",
    "Kuldeep Yadav",
    "Mohammed Shami",
    "MS Dhoni",
    "Ravindra Jadeja",
    "Rohit Sharma",
    "Shikhar Dhawan",
    "Vijay Shankar",
    "Virat Kohli",
    "Yuzvendra Chahal",
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()