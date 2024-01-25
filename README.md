## PoseCam📸

![Test Filter Camera](test_filter_camera.gif)

PoseCam is a `camera filter app` that integrated with a trained **model machine learning**.

With the integration of PoseCam and machine learning models, users can experience amazing selfie photography, where hand pose glasses stickers will automatically appear in real-time in face areas when users take selfies using **Pose V**, **Pose Metal**, or **Pose Thumb**. ✌️👍🤘

## Demo

https://readme.so/editor

## How To Use App

- Select the **Camera Filter✨** menu
- Click **START** button to use the webcam
- Take a selfie with a **V**, **Thumb**, or **Metal** hand pose
- Click **SNAPSHOT** button to take a picture
- Click **Download Photo** button to download the image taken

## Run Locally

Clone the project

```bash
  git clone https://github.com/ekawirawan/hand-pose-recognition
```

Go to the project directory

```bash
  cd hand-pose-recognition
```

Install dependencies

```bash
  pip install tensorflow
```

```bash
  pip install numpy
```

```bash
  pip install streamlit
```

```bash
  pip install twilio
```

```bash
  pip install opencv-python
```

```bash
  pip install firebase_admin
```

```bash
  pip install  av
```

```bash
  pip install firebase_admin
```

Get SID and secret token from **Twilio**

- Visit on https://www.twilio.com
- Log in, if you don’t have an account, Sign up.
- Create API keys & tokens on https://console.twilio.com
- Save your SID and secret token

Generate private key from **Firebase**

- Visit on https://firebase.google.com
- Log in, if you don’t have an account, Sign up.
- Click **Add project** button to add new project in your account
- Click add firebase to your web app
- Open **firebase Admin SDK** tab and generate new private key (it will be download .json file)
- Get started in storage
- Open setting rules change with code below:

```bash
  rules_version = '2';

  service firebase.storage {
    match /b/{bucket}/o {
        match /{allPaths=**} {
        allow read: if request.auth == null;
        allow write: if request.auth != null;
        }
    }
  }

```

Make secret management

- Create **.streamlit** directory

```bash
  mkdir .streamlit
```

- Create **secrets.toml** file in **.streamlit** directory

```bash
  cd .streamlit
  touch secrets.toml
```

- Add code below in secrets.toml file and change change accordingly your SID and auth token from **Twilio**

```bash
 TWILIO_ACCOUNT_SID = "..."
 TWILIO_AUTH_TOKEN = "..."
```

- Add code below in secrets.toml file and change change accordingly your .json file from **Firebase**

```bash
 [firebase]
 my_project_settings = {  "type" = "...",   "project_id"= "...",   "private_key_id"= "... ",   "private_key"= "...",   "client_email"= "...",   "client_id"= "...",   "auth_uri"= "...",   "token_uri"= "...",   "auth_provider_x509_cert_url"= "...",   "client_x509_cert_url"= "...", "universe_domain"= "..."}
```

Start the server

```bash
  streamlit run Home.py
```

## Tech Stack

### Streamlit

In developing the web-based PoseCam application, we chose to use **Streamlit** as the main framework, because Streamlit gives us ease and speed in building user interfaces using pure **Python**.

### Streamlit WebRTC

We utilized Streamlit WebRTC to activate the user's selfie camera directly on the web. By integrating **Tensorflow Lite models** with Streamlit WebRTC, our application can provide an interactive experience where users can engage their cameras to get accurate **camera filter** results

### Firebase

We use Firebase as a storage platform to store photo shots from our application users. With Firebase, users can **download** filtered photos quickly and easily.

### Tensorflow

We used TensorFlow to build and train a model, then implemented the model in TensorFlow Lite format to run efficiently in our web environment. TensorFlow Lite allows us to provide optimized machine learning model inference on the client side, allowing our web applications to run smoothly even with low web resources

---

## Machine Learning Model Selfie Hand Pose Recognition

### Purpose of Creating ML Models:

Our goal in creating a machine learning model is that our ML model will be able to differentiate or classify three hand poses that are commonly used when taking photos, namely: V pose, Thumb pose, and Metal pose.

### Expected results:

We expect the ML model to be able to classify hand poses with high accuracy, even under different lighting conditions and against different backgrounds. Apart from that, we also hope that the poses identified will be V Pose, Thumb Pose, and Metal Pose.

### Metrics Used In ML Model Design

#### Success Metrics:

Our Success Metrics are measured based on the model's precision in recognizing V, Thumb, and Metal poses using separate validation data.

#### Failure Metrics:

Our Failure metric is measured based on the model's inability to accurately detect hand pose, which can impact camera filter features and user experience.

### ML Model Assessment Criteria:

**Failure:** Our failure assessment criterion is when the ML model consistently fails to recognize commonly used hand poses when taking photos.

**Success:** Our success assessment criteria is when the ML model can provide a high level of precision in classifying hand poses with good accuracy.

### Model Output:

The output of our ML model is the result of classifying hand poses such as V Pose, Thumb Pose, or Metal Pose. This output will later be used to adjust the stickers on the user's camera display so that it can provide users with an exciting and enjoyable selfie experience.

### Implementation of Output From ML:

Our ML model output will be generated in real-time when users use our product on their device camera, whether in the form of a laptop, smartphone or tablet. In our product (selfie pose filter application), the output from this ML will later be used to display stickers according to the user's hand pose, the aim of which is to add a funny and fun impression when taking selfie photos.

### Below is some additional information in designing the Ml Selfie Hand Pose Recognition model.

#### Machine Learning Frameworks:

- TensorFlow

#### Web Application Frameworks:

- Streamlit

#### Programming language:

- Python

#### Supporting Libraries:

- NumPy
- Pandas
- Matplotlib
- openCV

#### Deployment Models:

- TensorFlow Serving

#### eployment Platforms:

- Streamlit Community Cloud

#### Version Control:

- Git
- GitHub

#### Development Environment:

- Google Colab
- VSCode

#### Evaluation Method:

- Precision

#### Hardware Resources:

- CPU
