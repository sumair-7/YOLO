import PIL.Image
import av
import cv2
import pandas as pd
import numpy as np
import streamlit as st
import sqlite3
import queue
import openpyxl
from streamlit_webrtc import (
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from src.api import load, get_model
import csv
import os.path

#database
conn = sqlite3.connect('database.db')
c = conn.cursor()
Flag = 0

def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS objectcount(item TEXT,itemcount INT,postdate DATE)')


create_table()


def add_data(item, itemcount, postdate):
    c.execute('INSERT INTO objectcount(item,itemcount,postdate) VALUES (?,?,?)',
              (item, itemcount, postdate))
    conn.commit()


def view_all():
    c.execute('SELECT * FROM objectcount')
    data = c.fetchall()
    return data

def csvformat(data):
    df = pd.DataFrame(data)
    df.to_csv('wine.csv')
    #st.write('Data is written successfully to csv File.') 

def excelformat(data):
    df = pd.DataFrame(data)
    #df.to_excel('wine.xlsx')
    # create excel writer object
    writer = pd.ExcelWriter('wine.xlsx')
    # write dataframe to excel
    df.to_excel(writer)
    # save the excel
    writer.save()
   

class VideoProcessor(VideoProcessorBase):
    """
    class for taking frame / sec and predict on it
    """
    def __init__(self):
        self.confidence_threshold = CONF_THR
        self.result_queue = queue.Queue()
        self.type = None

    def recv(self, frame):
        """
        :param frame: image array (height, width, channel)
        :return: image (height, width, channel) with bounding box
        """
        classes = ['8 PM',
                'Budweiser',      
                'Morpheus Brandy',       
                'Sula Wine']
        image_ = frame.to_ndarray(format="bgr24")
        img,counting_= load(model, image_, self.confidence_threshold, IMAGE_SIZE)
        C_ = {k: v for k, v in counting_.items() if v > 0}
        self.result_queue.put([C_])
        return av.VideoFrame.from_ndarray(img, format="bgr24")


@st.cache
def __model():
    """
    load model in cache mode
    Returns: torch.Module
    """
    return get_model('best3.pt')


if __name__ == '__main__':
    style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
    )  # permission for camera
    st.markdown(style, unsafe_allow_html=True)

    IMAGE_SIZE = 640  # default image size
    model = __model()  # model instance
    # change here for confidence of object predict in image
    # by default its 70
    CONF_THR = 0.70  # Confidence threshold

    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", 0.0, 1.0, CONF_THR, 0.05
    )  # Slide bar
    date = st.sidebar.date_input("Date")
    mode = st.sidebar.radio(
            "View Mode", ('🎥 video', '🖼️ image', '📊 data'))
    if mode == '🎥 video':
        button_placeholder = st.empty()
        st.title("🎥 Wine Detection Live Video")
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            client_settings=WEBRTC_CLIENT_SETTINGS,
            video_processor_factory=VideoProcessor,
            async_processing=True,
        )
        # if st.button('Capture'):
        #     Flag += 1
        if webrtc_ctx.video_processor:
            # checks if camera is running
            webrtc_ctx.video_processor.confidence_threshold = confidence_threshold
        if st.checkbox("Store", value=False):
            Flag = 1
        # if st.checkbox("Stop it", value=False):
        #     Flag = 0
        if st.checkbox("Show the detected labels", value=True):
            if webrtc_ctx.state.playing:
                labels_placeholder = st.empty()
                # button_placeholder = st.empty()
                empty = st.empty()
                while True:
                    if webrtc_ctx.video_processor:
                        # webrtc_ctx.video_processor.type = st.radio(
                        #     "Capture", ("No", "Yes")
                        #     )
                        try:
                            result = webrtc_ctx.video_processor.result_queue.get(
                                timeout=1.0
                            )
                        except queue.Empty:
                            result = None
                        if result:
                            data_ = pd.DataFrame(result[0], index=['items'])
                            labels_placeholder.table(data_)
                            if Flag:
                                for name, d in result[0].items():
                                    add_data(name, d, date)
                                Flag = 0
                                # Flag -= 1
                        else:
                            labels_placeholder.table(result)
                    else:
                        break
    elif mode == '🖼️ image':
        st.title("🖼️ Wine Bottle detection image")
        img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if img_file_buffer is not None:
            image = np.array(PIL.Image.open(img_file_buffer))  # Open buffer
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))  # resize image
            image_box,counting = load(model, image, confidence_threshold, IMAGE_SIZE)  # function to predict on image
            st.image(
                    image_box, caption=f"Processed image", use_column_width=True,
                )
            C = {k: v for k, v in counting.items() if v > 0}
            data = pd.DataFrame(C, index=['items'])
            st.sidebar.table(data)
            for name, d in C.items():
                    add_data(name, d, date)
    elif mode == '📊 data':
            st.title("📊 data")
            ALL_DATA = view_all()
            # for i in ALL_DATA[::-1]:
            #     st.markdown(head_message_temp.format(i[0], i[1], i[2]), unsafe_allow_html=True)
            one = [i[0] for i in ALL_DATA[::-1]]
            two = [i[1] for i in ALL_DATA[::-1]]
            three = [i[2] for i in ALL_DATA[::-1]]
            DATA = {
                'Time': three,
                'Name': one,
                'Count': two
            }
            data_ = pd.DataFrame.from_dict(DATA)
            st.table(data_)
            stat = data_.groupby('Name')['Count'].sum()
            st.table(pd.DataFrame(stat))
            Downloadmode = st.sidebar.radio(
            "Download Mode", ('None','Excel', 'CSV'))
            if Downloadmode=='CSV':
                 csvformat(data_)
                 st.write('Data is written successfully to csv File.')

            elif Downloadmode=='Excel':
                excelformat(data_)
                st.write('Data is written successfully to Excel File.')
           