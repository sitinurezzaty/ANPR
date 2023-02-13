import streamlit as st
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import tensorflow as tf
#import tensorflow_hub as hub
import time ,sys
from streamlit_embedcode import github_gist
import urllib.request
import urllib
import moviepy.editor as moviepy
from matplotlib import pyplot as plt

#library for character recognition
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder

#library for send email
import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def object_detection_video(): 
  
    #function for character recognition (start)
    #contour cropped_img
    def get_contour_precedence(contour, cols):
        tolerance_factor = 10
        origin = cv2.boundingRect(contour)
        return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]
    
    #resize cropped_img
    def square(img):
        """
        This function resize non square image to square one (height == width)
        :param img: input image as numpy array
        :return: numpy array
        """
        
        # image after making height equal to width
        squared_image = img
        
        # Get image height and width
        h = img.shape[0]
        w = img.shape[1]
        
        # In case height superior than width
        if h > w:
            diff = h-w
            if diff % 2 == 0:
                x1 = np.zeros(shape=(h, diff//2))
                x2 = x1
            else:
                x1 = np.zeros(shape=(h, diff//2))
                x2 = np.zeros(shape=(h, (diff//2)+1))

            squared_image = np.concatenate((x1, img, x2), axis=1)
            
         # In case height inferior than width
        if h < w:
            diff = w-h
            if diff % 2 == 0:
                x1 = np.zeros(shape=(diff//2, w))
                x2 = x1
            else:
                x1 = np.zeros(shape=(diff//2, w))
                x2 = np.zeros(shape=((diff//2)+1, w))
            
            squared_image = np.concatenate((x1, img, x2), axis=0)

        return squared_image
    
    def sort(vector):
        sort = True
        while (sort == True):
            
            sort = False
            for i in range(len(vector) - 1):
                x_1 = vector[i][0]
                y_1 = vector[i][1]
                
                for j in range(i + 1, len(vector)):
                    x_2 = vector[j][0]
                    y_2 = vector[j][1]
                    
                    if (x_1 >= x_2 and y_2 >= y_1):
                        tmp = vector[i]
                        vector[i] = vector[j]
                        vector[j] = tmp
                        sort = True
                    
                    elif (x_1 < x_2 and y_2 > y_1):
                        tmp = vector[i]
                        vector[i] = vector[j]
                        vector[j] = tmp
                        sort = True
                        
        return vector
                
        
    def plate_segmentation(img):
        #img = cv2.imread(img)
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        height = img.shape[0]
        width = img.shape[1]
        area = height * width
        
        #scale1 = 0.001
        scale1 = 0.01
        scale2 = 0.1
        area_condition1 = area * scale1
        area_condition2 = area * scale2
        # global thresholding
        ret1,th1 = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
        
        # Otsu's thresholding
        ret2,th2 = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(imgray,(5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # sort contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        cropped = dict()
        #cropped = []
        for cnt in contours:
            (x,y,w,h) = cv2.boundingRect(cnt)
            distance_center = (2*x+w)/2
            
            if distance_center in cropped:
                pass
            else:            
                if (w * h > area_condition1 and w * h < area_condition2 and w/h > 0.3 and h/w > 1):
                    cv2.drawContours(img, [cnt], 0, (0, 255, 0), 1)
                    cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 1)
                    c = th2[y:y+h,x:x+w]
                    c = np.array(c)
                    c = cv2.bitwise_not(c)
                    c = square(c)
                    c = cv2.resize(c,(28,28), interpolation = cv2.INTER_AREA)
                    cropped[distance_center] = c
                    #cropped.append(c)
                    
        sorted_cropped = []
        for x_center in sorted(cropped):
            sorted_cropped.append(cropped[x_center])
            
        cv2.imwrite('detection.png', img)
        return img, sorted_cropped 



    #object_detection_video.has_beenCalled = True
    #pass
    CONFIDENCE = 0.5
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    config_path = r'model/yolov4-custom_64_16_0001.cfg'
    weights_path = r'model/yolov4-custom_64_16_0001_best.weights'
    font_scale = 1
    thickness = 1
    url = "https://github.com/sitinurezzaty/ANPR/blob/main/names.txt"
    f = urllib.request.urlopen(url)
    labels = [line.decode('utf-8').strip() for  line in f]
    #f = open(r'C:\Users\Olazaah\Downloads\stream\labels\coconames.txt','r')
    #lines = f.readlines()
    #labels = [line.strip() for line in lines]
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    st.title("Car Plate Recognition (CPR)")
    st.subheader("""
    This system only receive mp4, MOV, and MPEG format
    """
    )
    uploaded_video = st.file_uploader("Upload Video", type = ['mp4','mpeg','mov'])
    if uploaded_video != None:
        
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) # save video to disk

        st_video = open(vid,'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")
        
        #to get video capture object
        cap = cv2.VideoCapture(vid)
        
        raw_data = []
        
        #it will turn TRUE, if the frames was read correctly
        _, image = cap.read()
        #extract the rows and columns values from the shape tuple
        h, w = image.shape[:2]
        #write video to the disk bfr do detection
        fourcc = cv2.VideoWriter_fourcc(*'mpv4')
        out = cv2.VideoWriter("detected_video.mp4", fourcc, 20.0, (w, h))
        count = 0
        #loop over frames from video stream
        while True:    
            _, image = cap.read()
            
            # if no frame is grabbed, we reached the end of video, so break the loop
            #if not _:
                #break
                
            # if the frame dimensions are empty, grab them
            #if w is None or h is None:
            #    (h,w) = frame.shape[:2]
                
            if _ != False:
                h, w = image.shape[:2]
                blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                start = time.perf_counter()
                layer_outputs = net.forward(ln)
                time_took = time.perf_counter() - start
                count +=1
                print(f"Time took: {count}", time_took)
                boxes, confidences, class_ids = [], [], []

                # loop over each of the layer outputs
                for output in layer_outputs:
                    # loop over each of the object detections
                    for detection in output:
                        # extract the class id (label) and confidence (as a probability) of
                        # the current object detection
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        # discard weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > CONFIDENCE:
                            # scale the bounding box coordinates back relative to the
                            # size of the image, keeping in mind that YOLO actually
                            # returns the center (x, y)-coordinates of the bounding
                            # box followed by the boxes' width and height
                            box = detection[:4] * np.array([w, h, w, h])
                            (centerX, centerY, width, height) = box.astype("int")

                            # use the center (x, y)-coordinates to derive the top and
                            # and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # update our list of bounding box coordinates, confidences,
                            # and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                # perform the non maximum suppression given the scores defined before
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

                font_scale = 0.8
                thickness = 2

                # ensure at least one detection exists
                if len(idxs) > 0:
                    # loop over the indexes we are keeping
                    for i in idxs.flatten():
                        # extract the bounding box coordinates
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]
                        # draw a bounding box rectangle and label on the image
                        color = [int(c) for c in colors[class_ids[i]]]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
                        # calculate text width & height to draw the transparent boxes as background of the text
                        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
                        text_offset_x = x
                        text_offset_y = y - 5
                        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                        overlay = image.copy()
                        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                        # add opacity (transparency to the box)
                        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                        # now put the text (label: confidence %)
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
                        
                        #try cropped img
                        model = keras.models.load_model('model/cnn_classifier.h5')
                        #just cropped image of number plate je
                        cropped_img=image[y:y+h,x:x+w] #print out image berjaya
                        print(type(cropped_img))
                        
                        #patutnya kat sini ada if statement ()
                        img_char, digits = plate_segmentation(cropped_img)
                        #st.image(img_char) #to print image
                        
                        
                        #function for detecting the character
                        # Predicting the output
                        def fix_dimension(img_char): 
                            new_img = np.zeros((28,28,3))
                            for i in range(3):
                                new_img[:,:,i] = img_char
                            return new_img

                        output = []
                        for d in digits:
                            img_ = cv2.resize(d, (28,28), interpolation=cv2.INTER_AREA)
                            img = fix_dimension(img_)
                            d = img.reshape(1,28,28,3) #preparing image for the model
                            #d = np.reshape(d, (1, 28,28, 1))
                            outc = model.predict(d)
                            # Get max pre arg
                            p = []
                            precision = 0
                            for i in range(len(outc)):
                                z = np.zeros(36)
                                z[np.argmax(outc[i])] = 1.
                                precision = max(outc[i])
                                p.append(z)
                            prediction = np.array(p)

                            # one hot encoding
                            alphabets = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
                            classes = []
                            for a in alphabets:
                                classes.append([a])
                            ohe = OneHotEncoder(handle_unknown='ignore')
                            ohe.fit(classes)
                            pred = ohe.inverse_transform(prediction)

                            #st.write('Prediction : ' + str(pred[0][0]) + ' , Precision : ' + str(precision))
                            #try print out the output in a str form 
                            output.append(str(pred[0][0]))
                            plate_number = ''.join(map(str, output))



                            #Data is stored in CSV file
                            raw_data.append({'Date': [time.asctime( time.localtime(time.time()))], 'Number Plate': [plate_number]})

                            df = pd.DataFrame(raw_data, columns = ['Date', 'Number Plate'])
                            df.to_csv('Traffic_Notice.csv')
                        

                out.write(image)
                cv2.imshow("image", image)
                
                if ord("q") == cv2.waitKey(1):
                    break
            else:
                break


        #return "detected_video.mp4"
            
        cap.release()
        cv2.destroyAllWindows()
        
        #call function send_email()
        send_email()    

        
#send_email function
def send_email():
    subject = "TRAFFIC VIOLATION NOTICE"
    body = """
    <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
        <title>Simple Transactional Email</title>
        <style>
          /* -------------------------------------
              GLOBAL RESETS
          ------------------------------------- */

          /*All the styling goes here*/

          img {
            border: none;
            -ms-interpolation-mode: bicubic;
            max-width: 100%; 
          }

          body {
            background-color: #f6f6f6;
            font-family: sans-serif;
            -webkit-font-smoothing: antialiased;
            font-size: 14px;
            line-height: 1.4;
            margin: 0;
            padding: 0;
            -ms-text-size-adjust: 100%;
            -webkit-text-size-adjust: 100%; 
          }

          table {
            border-collapse: separate;
            mso-table-lspace: 0pt;
            mso-table-rspace: 0pt;
            width: 100%; }
            table td {
              font-family: sans-serif;
              font-size: 14px;
              vertical-align: top; 
          }

          /* -------------------------------------
              BODY & CONTAINER
          ------------------------------------- */

          .body {
            background-color: #f6f6f6;
            width: 100%; 
          }

          /* Set a max-width, and make it display as block so it will automatically stretch to that width, but will also shrink down on a phone or something */
          .container {
            display: block;
            margin: 0 auto !important;
            /* makes it centered */
            max-width: 580px;
            padding: 10px;
            width: 580px; 
          }

          /* This should also be a block element, so that it will fill 100% of the .container */
          .content {
            box-sizing: border-box;
            display: block;
            margin: 0 auto;
            max-width: 580px;
            padding: 10px; 
          }

          /* -------------------------------------
              HEADER, FOOTER, MAIN
          ------------------------------------- */
          .main {
            background: #ffffff;
            border-radius: 3px;
            width: 100%; 
          }

          .wrapper {
            box-sizing: border-box;
            padding: 20px; 
          }

          .content-block {
            padding-bottom: 10px;
            padding-top: 10px;
          }

          .footer {
            clear: both;
            margin-top: 10px;
            text-align: center;
            width: 100%; 
          }
            .footer td,
            .footer p,
            .footer span,
            .footer a {
              color: #999999;
              font-size: 12px;
              text-align: center; 
          }

          /* -------------------------------------
              TYPOGRAPHY
          ------------------------------------- */
          h1,
          h2,
          h3,
          h4 {
            color: #000000;
            font-family: sans-serif;
            font-weight: 400;
            line-height: 1.4;
            margin: 0;
            margin-bottom: 30px; 
          }

          h1 {
            font-size: 35px;
            font-weight: 300;
            text-align: center;
            text-transform: capitalize; 
          }

          p,
          ul,
          ol {
            font-family: sans-serif;
            font-size: 14px;
            font-weight: normal;
            margin: 0;
            margin-bottom: 15px; 
          }
            p li,
            ul li,
            ol li {
              list-style-position: inside;
              margin-left: 5px; 
          }

          a {
            color: #3498db;
            text-decoration: underline; 
          }

          /* -------------------------------------
              BUTTONS
          ------------------------------------- */
          .btn {
            box-sizing: border-box;
            width: 100%; }
            .btn > tbody > tr > td {
              padding-bottom: 15px; }
            .btn table {
              width: auto; 
          }
            .btn table td {
              background-color: #ffffff;
              border-radius: 5px;
              text-align: center; 
          }
            .btn a {
              background-color: #ffffff;
              border: solid 1px #3498db;
              border-radius: 5px;
              box-sizing: border-box;
              color: #3498db;
              cursor: pointer;
              display: inline-block;
              font-size: 14px;
              font-weight: bold;
              margin: 0;
              padding: 12px 25px;
              text-decoration: none;
              text-transform: capitalize; 
          }

          .btn-primary table td {
            background-color: #3498db; 
          }

          .btn-primary a {
            background-color: #3498db;
            border-color: #3498db;
            color: #ffffff; 
          }

          /* -------------------------------------
              OTHER STYLES THAT MIGHT BE USEFUL
          ------------------------------------- */
          .last {
            margin-bottom: 0; 
          }

          .first {
            margin-top: 0; 
          }

          .align-center {
            text-align: center; 
          }

          .align-right {
            text-align: right; 
          }

          .align-left {
            text-align: left; 
          }

          .clear {
            clear: both; 
          }

          .mt0 {
            margin-top: 0; 
          }

          .mb0 {
            margin-bottom: 0; 
          }

          .preheader {
            color: transparent;
            display: none;
            height: 0;
            max-height: 0;
            max-width: 0;
            opacity: 0;
            overflow: hidden;
            mso-hide: all;
            visibility: hidden;
            width: 0; 
          }

          .powered-by a {
            text-decoration: none; 
          }

          hr {
            border: 0;
            border-bottom: 1px solid #f6f6f6;
            margin: 20px 0; 
          }

          /* -------------------------------------
              RESPONSIVE AND MOBILE FRIENDLY STYLES
          ------------------------------------- */
          @media only screen and (max-width: 620px) {
            table.body h1 {
              font-size: 28px !important;
              margin-bottom: 10px !important; 
            }
            table.body p,
            table.body ul,
            table.body ol,
            table.body td,
            table.body span,
            table.body a {
              font-size: 16px !important; 
            }
            table.body .wrapper,
            table.body .article {
              padding: 10px !important; 
            }
            table.body .content {
              padding: 0 !important; 
            }
            table.body .container {
              padding: 0 !important;
              width: 100% !important; 
            }
            table.body .main {
              border-left-width: 0 !important;
              border-radius: 0 !important;
              border-right-width: 0 !important; 
            }
            table.body .btn table {
              width: 100% !important; 
            }
            table.body .btn a {
              width: 100% !important; 
            }
            table.body .img-responsive {
              height: auto !important;
              max-width: 100% !important;
              width: auto !important; 
            }
          }

          /* -------------------------------------
              PRESERVE THESE STYLES IN THE HEAD
          ------------------------------------- */
          @media all {
            .ExternalClass {
              width: 100%; 
            }
            .ExternalClass,
            .ExternalClass p,
            .ExternalClass span,
            .ExternalClass font,
            .ExternalClass td,
            .ExternalClass div {
              line-height: 100%; 
            }
            .apple-link a {
              color: inherit !important;
              font-family: inherit !important;
              font-size: inherit !important;
              font-weight: inherit !important;
              line-height: inherit !important;
              text-decoration: none !important; 
            }
            #MessageViewBody a {
              color: inherit;
              text-decoration: none;
              font-size: inherit;
              font-family: inherit;
              font-weight: inherit;
              line-height: inherit;
            }
            .btn-primary table td:hover {
              background-color: #34495e !important; 
            }
            .btn-primary a:hover {
              background-color: #34495e !important;
              border-color: #34495e !important; 
            } 
          }

        </style>
      </head>
      <body>
        <span class="preheader">This is preheader text. Some clients will show this text as a preview.</span>
        <table role="presentation" border="0" cellpadding="0" cellspacing="0" class="body">
          <tr>
            <td>&nbsp;</td>
            <td class="container">
              <div class="content">

                <!-- START CENTERED WHITE CONTAINER -->
                <table role="presentation" class="main">

                  <!-- START MAIN CONTENT AREA -->
                  <tr>
                    <td class="wrapper">
                      <table role="presentation" border="0" cellpadding="0" cellspacing="0">
                        <tr>
                          <td>
                            <p>Tuan/Puan/Sir/Madam</p>
                            <p style="color:red;">-------PLEASE DO NOT REPLY TO THIS EMAIL. THIS MAILBOX IS NOT MONITORED AND YOU WILL NOT RECEIVE A RESPONSE-------</p>
                            <p>Dimaklumkan bahawa Tuan/Puan/Cik/Encik telah melakukan kesalahan lalulintas. Butiran kesalahan anda adalah seperti yang ditunjukkan di dalam fail di bawah. Tuan/Puan/Cik/Encik boleh memeriksa perincian kesalahan lalu lintas di dalam fail yang dilampirkan dalam emel ini (data.csv). <br><i>To be informed that you have committed a traffic offense. The details of your fault are as shown in the file below. Sir/Madam/Ms./Mr., you can check the details of traffic offences in the file attached to this email (data.csv).</i></p><br>
                            
                            <p>Mohon Tuan/Puan/Cik/Encik menjelaskan bayaran saman kesalahan jalan lalu lintas yang dilakukan sebelum 24jam. <br><i>Sir/Madam/Ms./Mr., please pay the summons for the traffic offence that has been committed.</i></p><br>
                            <table role="presentation" border="0" cellpadding="0" cellspacing="0" class="btn btn-primary">
                              <tbody>
                                <tr>
                                  <td align="left">
                                    <table role="presentation" border="0" cellpadding="0" cellspacing="0">
                                      <tbody>
                                        <tr>
                                          <td> <a href="http://htmlemail.io" target="_blank">Link</a> </td>
                                        </tr>
                                      </tbody>
                                    </table>
                                  </td>
                                </tr>
                              </tbody>
                            </table>
                          </td>
                        </tr>
                      </table>
                    </td>
                  </tr>

                <!-- END MAIN CONTENT AREA -->
                </table>
                <!-- END CENTERED WHITE CONTAINER -->

              </div>
            </td>
            <td>&nbsp;</td>
          </tr>
        </table>
      </body>
    </html>    
    """
    
    sender_email = "stnurezzaty@gmail.com"
    receiver_email = "sitinurezzaty95@gmail.com"
    password = "kjjmaoqoqhtuiwph"

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message["Bcc"] = receiver_email  # Recommended for mass emails

    # Add body to email
    message.attach(MIMEText(body, "html"))

    filename = "Traffic_Notice.csv"  # In same directory as script

    # Open PDF file in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode file in ASCII characters to send by email    
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )

    # Add attachment to message and convert message to string
    message.attach(part)
    text = message.as_string()

    # Log in to server using secure context and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)





def main():
    
    #styling streamlit app
    st.set_page_config(page_title="Car Plate Recognition (CPR) App")
    
    
    new_title = '<h1 style="font-size: 42px">Car PLate Recognition (CPR) System</h1>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    # read_me = st.markdown("""
    #<p style="font-size: 18px;">Number plate recognition system specifically designed to read the vehicle number plate that used You-Only-Look-Once (YOLOv4) and Convolutional Neural Network (CNN)</p>"""
    #)
    content = '<p style="font-size: 18px; align:justify">Car plate recognition (CPR) systems are technologies that aid Malaysian law enforcement in handling or issuing summonses.This system is designed to be able to detect and recognise the vehicle plate number using You-Only-Look-Once (YOLOv4) and Convolutional Neural Network (CNN). With the ability of the system to integrate with email, it will be able to reduce the traffic violations while also transitioning to a paperless society.</p>'
    #)
    read_me = st.markdown(content, unsafe_allow_html=True)
    
    st.sidebar.title("Select Activity")
    choice  = st.sidebar.selectbox("MODE",("About", "Video Streaming"))
    #["Show Instruction","Landmark identification","Show the #source code", "About"]
    
            
    if choice == "Video Streaming":
        read_me_0.empty()
        read_me.empty()
        #object_detection_video.has_beenCalled = False
        object_detection_video()
       
        
        #if object_detection_video.has_beenCalled:
        try:

            clip = moviepy.VideoFileClip('detected_video.mp4')
            clip.write_videofile("myvideo.mp4")
            st_video = open('myvideo.mp4','rb')
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Detected Video") 
        except OSError:
            ''
        
    elif choice == "About":
        print()
        

if __name__ == '__main__':
		main()	
