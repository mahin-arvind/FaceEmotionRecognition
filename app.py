import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# load model
emotion_dict = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

classifier =load_model("vgg_model.h5")
classifier2 =load_model("custom_model.h5")

# load weights into new model
classifier.load_weights("vgg_model.h5")
classifier2.load_weights("custom_model.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer_1(VideoTransformerBase): #VGG MODEL
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = 'VGG: ' + str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

class VideoTransformer_2(VideoTransformerBase):#CONV MODEL
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier2.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = 'CONV: ' + str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # Face Analysis Application #
    st.title("Real-Time Face Emotion Detection")
    activities = ["Home","About"]
    choice = st.sidebar.selectbox("MENU", activities)
  
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:tomato";padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Start Your Real Time Face Emotion Detection.</h4>
                                            </div>
                                            </br>"""

        st.markdown(html_temp_home1, unsafe_allow_html=True)

        model_select =  st.selectbox("Select Model",["None","VGG", "CONV"])
  
        if model_select == "VGG":    
            st.subheader("VGG Live Feed")
            st.write("1. Hit Start and enable camera permission.")
            st.write("2. Hit Stop to end demo")
            st.write("3. Try Different Models only after stopping present demo.")
            webrtc_streamer(key="example", video_processor_factory=VideoTransformer_1)
            
            st.subheader("Model Information")
            st.write("Recall: 65.1 % ")
            st.write("Precision: 65.3 % ")
            st.write("F1 Score: 65 %")
            st.write("Balanced Accuracy: 62.2 % ")

        if model_select== "CONV":
            st.subheader("CONV Live Feed")
            st.write("1. Hit Start and enable camera permission.")
            st.write("2. Hit Stop to end demo")
            st.write("3. Try Different Models only after stopping present demo.")
            webrtc_streamer(key="example", video_processor_factory=VideoTransformer_2)
            
            st.subheader("Model Information")
            st.write("Recall: 64.7 %")
            st.write("Precision: 64.9 %")
            st.write("F1 Score: 64.3 %") 
            st.write("Balanced Accuracy: 61.2 %")

        elif model_select == "None":
            st.info("What's cooking, good looking? Go ahead and pick a model!  ")
            st.caption("Thanks for Visiting!")
        else:
            pass
    
    elif choice == "About":

        st.subheader("About the Application")
        html_temp4 = """
                                    <div style="background-color:tomato;padding:10px">
                                    <h4 style="color:white;text-align:center;">This application provides three models to perform 
                                    realtime face emotion recognition.</div>
                                    <div>They are: Custom VGG Blocks and Custom Conv Layers.
                                    Face Emotion Recognition 2013 was used to train these models </div>
                                    <br></br>
                                    <br></br>"""
        st.markdown(html_temp4, unsafe_allow_html=True)

                              
        st.subheader("""Email""")
        st.info(""">Mahin : mahinarvindds@gmail.com
                    """)

        st.subheader("""Project Repository""")     
        st.info(""">Mahin Arvind (https://github.com/mahin-arvind/Face-Emotion-Recognition)
                    """)

    else:
        pass

    # Set footer
    footer="""<style> a:link , a:visited{color: blue;background-color: transparent;text-decoration: underline;} 
    a:hover,  a:active {color: red;background-color: transparent;text-decoration: underline;}
    .footer {position: fixed;left: 0;bottom: 0;width: 100%;background-color: white;color: black;text-align: center;}
    </style><div class="footer"><p></p> </div>"""
    st.markdown(footer,unsafe_allow_html=True)


if __name__ == "__main__":
    main()
