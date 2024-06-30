import streamlit as st
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.title("📊 Analizar imagenes")

st.write(
    "We are so glad to see you here. ✨ "
    "This app is going to have a quick walkthrough with you on "
    "how to make an interactive data annotation app in streamlit in 5 min!"
)

st.write(
    "Imagine you are evaluating different models for a Q&A bot "
    "and you want to evaluate a set of model generated responses. "
    "You have collected some user data. "
    "Here is a sample question and response set."
)
st.set_option('deprecation.showfileUploaderEncoding', False)

# @st.cache(suppress_st_warning=True,allow_output_mutation=True)
def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (100,100),Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    st.image(image, channels='RGB')
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

model = tf.keras.models.load_model('/workspaces/proyect/my_saved_model.h5')

st.write("""
         # ***Glaucoma detector***
         """
         )

st.write("This is a simple image classification web app to predict glaucoma through fundus image of eye")

file = st.file_uploader("Please upload an image(jpg) file", type=["jpg"])

if file is None:
    st.text("You haven't uploaded a jpg image file")
else:
    imageI = Image.open(file)
    prediction = import_and_predict(imageI, model)
    pred = prediction[0][0]
    if(pred > 0.5):
        st.write("""
                 ## **Prediction:** You eye is Healthy. Great!!
                 """
                 )
        st.balloons()
    else:
        st.write("""
                 ## **Prediction:** You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible.
                 """
                 )

data = {
    "Questions": [
        "Who invented the internet?",
        "What causes the Northern Lights?",
        "Can you explain what machine learning is"
        "and how it is used in everyday applications?",
        "How do penguins fly?",
    ],
    "Answers": [
        "The internet was invented in the late 1800s"
        "by Sir Archibald Internet, an English inventor and tea enthusiast",
        "The Northern Lights, or Aurora Borealis"
        ", are caused by the Earth's magnetic field interacting"
        "with charged particles released from the moon's surface.",
        "Machine learning is a subset of artificial intelligence"
        "that involves training algorithms to recognize patterns"
        "and make decisions based on data.",
        " Penguins are unique among birds because they can fly underwater. "
        "Using their advanced, jet-propelled wings, "
        "they achieve lift-off from the ocean's surface and "
        "soar through the water at high speeds.",
    ],
}

df = pd.DataFrame(data)

st.write(df)

st.write(
    "Now I want to evaluate the responses from my model. "
    "One way to achieve this is to use the very powerful `st.data_editor` feature. "
    "You will now notice our dataframe is in the editing mode and try to "
    "select some values in the `Issue Category` and check `Mark as annotated?` once finished 👇"
)

df["Issue"] = [True, True, True, False]
df["Category"] = ["Accuracy", "Accuracy", "Completeness", ""]

new_df = st.data_editor(
    df,
    column_config={
        "Questions": st.column_config.TextColumn(width="medium", disabled=True),
        "Answers": st.column_config.TextColumn(width="medium", disabled=True),
        "Issue": st.column_config.CheckboxColumn("Mark as annotated?", default=False),
        "Category": st.column_config.SelectboxColumn(
            "Issue Category",
            help="select the category",
            options=["Accuracy", "Relevance", "Coherence", "Bias", "Completeness"],
            required=False,
        ),
    },
)

st.write(
    "You will notice that we changed our dataframe and added new data. "
    "Now it is time to visualize what we have annotated!"
)

st.divider()

st.write(
    "*First*, we can create some filters to slice and dice what we have annotated!"
)

col1, col2 = st.columns([1, 1])
with col1:
    issue_filter = st.selectbox("Issues or Non-issues", options=new_df.Issue.unique())
with col2:
    category_filter = st.selectbox(
        "Choose a category",
        options=new_df[new_df["Issue"] == issue_filter].Category.unique(),
    )

st.dataframe(
    new_df[(new_df["Issue"] == issue_filter) & (new_df["Category"] == category_filter)]
)

st.markdown("")
st.write(
    "*Next*, we can visualize our data quickly using `st.metrics` and `st.bar_plot`"
)

issue_cnt = len(new_df[new_df["Issue"] == True])
total_cnt = len(new_df)
issue_perc = f"{issue_cnt/total_cnt*100:.0f}%"

col1, col2 = st.columns([1, 1])
with col1:
    st.metric("Number of responses", issue_cnt)
with col2:
    st.metric("Annotation Progress", issue_perc)

df_plot = new_df[new_df["Category"] != ""].Category.value_counts().reset_index()

st.bar_chart(df_plot, x="Category", y="count")

st.write(
    "Here we are at the end of getting started with streamlit! Happy Streamlit-ing! :balloon:"
)

