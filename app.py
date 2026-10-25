
import os
import uuid
import sqlite3

import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import qrcode
from io import BytesIO
import streamlit as st
from supabase import create_client


# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="FindIt AI",
    page_icon="🔎",
    layout="wide"
)


# --------------------------------------------------
# 2. Directories and Database
# --------------------------------------------------

os.makedirs("item_images", exist_ok=True)

# Connect to Supabase
supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)

connection = sqlite3.connect("findit.db")
cursor = connection.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS found_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_name TEXT NOT NULL,
        description TEXT NOT NULL,
        location TEXT NOT NULL
    )
""")

connection.commit()

# Add image_path column for existing databases
cursor.execute("PRAGMA table_info(found_items)")
columns = [column[1] for column in cursor.fetchall()]

if "image_path" not in columns:
    cursor.execute(
        "ALTER TABLE found_items ADD COLUMN image_path TEXT"
    )
    connection.commit()


# --------------------------------------------------
# 3. Load CLIP AI Model
# --------------------------------------------------

@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    model.eval()

    return model, processor


with st.spinner("Loading AI model..."):
    clip_model, clip_processor = load_clip_model()


# --------------------------------------------------
# 4. AI Image Similarity Function
# --------------------------------------------------

def get_image_features(images):
    inputs = clip_processor(
        images=images,
        return_tensors="pt"
    )

    with torch.no_grad():
        vision_output = clip_model.vision_model(**inputs)

        features = clip_model.visual_projection(
            vision_output.pooler_output
        )

    # Normalize image features
    features = features / features.norm(
        dim=-1,
        keepdim=True
    ).clamp(min=1e-12)

    return features


def find_similar_items(query_image, image_items):
    if not image_items:
        return []

    saved_images = [
        saved_image
        for item, saved_image in image_items
    ]

    query_features = get_image_features(query_image)
    database_features = get_image_features(saved_images)

    scores = (
        query_features @ database_features.T
    ).squeeze(0)

    ranked_items = sorted(
        zip(image_items, scores.tolist()),
        key=lambda result: result[1],
        reverse=True
    )

    return ranked_items


# --------------------------------------------------
# 5. Website Heading
# --------------------------------------------------

st.title("🔎 FindIt AI")
st.header("📱 Share FindIt AI")

website_url = "https://ai-powered-lost-and-found-system-using-image-similarity-search.streamlit.app/"

qr = qrcode.QRCode(box_size=10, border=4)
qr.add_data(website_url)
qr.make(fit=True)

qr_image = qr.make_image(fill_color="black", back_color="white")

qr_buffer = BytesIO()
qr_image.save(qr_buffer, format="PNG")
qr_bytes = qr_buffer.getvalue()

st.image(qr_bytes, caption="Scan to open FindIt AI", width=250)

st.download_button(
    label="⬇️ Download QR Code",
    data=qr_bytes,
    file_name="FindItAI_QR_Code.png",
    mime="image/png"
)
st.subheader("Intelligent Lost-and-Found System")

st.write(
    "Lost something on campus? Let's help you find it!"
)

st.divider()


# --------------------------------------------------
# 6. Report a Found Item
# --------------------------------------------------

st.header("📦 Report a Found Item")

with st.form("report_item_form"):

    uploaded_image = st.file_uploader(
        "Upload a photo of the found item",
        type=["jpg", "jpeg", "png"],
        key="report_image"
    )   
    camera_image = st.camera_input(
        "Or take a photo using your camera",
        key="report_camera"
    )

    if camera_image is not None:
        uploaded_image = camera_image
    

    item_name = st.text_input(
        "What item did you find?"
    )

    description = st.text_area(
        "Describe the item"
    )

    location = st.text_input(
        "Where did you find it?"
    )

    submit_item = st.form_submit_button(
        "Submit Found Item"
    )


if submit_item:

    if (
        item_name.strip()
        and description.strip()
        and location.strip()
    ):

        image_path = None

        if uploaded_image is not None:

            extension = os.path.splitext(
                uploaded_image.name
            )[1].lower()

            image_filename = (
                f"{uuid.uuid4().hex}{extension}"
            )

            image_bytes = uploaded_image.getvalue()

            supabase.storage.from_("item-images").upload(
                path=image_filename,
                file=image_bytes,
                file_options={
                    "content-type": uploaded_image.type,
                    "upsert": "false"
                }
            )

            image_path = supabase.storage.from_("item-images").get_public_url(
                image_filename
            )

        supabase.table("found_items").insert({
            "item_name": item_name.strip(),
            "description": description.strip(),
            "location": location.strip(),
            "image_path": image_path
        }).execute()

        st.success("Found item submitted successfully!")

    else:
        st.error("Please fill in all required fields.")
        
# Create a table to store item claims
cursor.execute("""
    CREATE TABLE IF NOT EXISTS claims (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_id INTEGER NOT NULL,
        claimant_name TEXT NOT NULL,
        contact TEXT NOT NULL,
        ownership_proof TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (item_id) REFERENCES found_items(id)
    )
""")

connection.commit()

st.success("Found item saved successfully!")


st.divider()
# --------------------------------------------------
# 7. AI Image Search
# --------------------------------------------------

st.header("🔎 Search Found Items Using AI")

ai_search_image = st.file_uploader(
    "Upload an image to find similar items",
    type=["jpg", "jpeg", "png"],
    key="ai_search_image"
)

camera_search_image = st.camera_input(
    "Or take a photo to search for similar items",
    key="camera_search_image"
)

if camera_search_image is not None:
    ai_search_image = camera_search_image

if ai_search_image is not None:

    try:
        query_image = Image.open(
            ai_search_image
        ).convert("RGB")

        st.write("**Your uploaded search image:**")
        st.image(query_image, width=220)

        response = (
            supabase.table("found_items")
            .select("*")
            .not_.is_("image_path", "null")
            .order("id", desc=True)
            .execute()
        )

        database_items = response.data
        image_items = []

        import requests
        from io import BytesIO

        for item in database_items:

            saved_url = item.get("image_path")

            if saved_url and saved_url.startswith("https://"):

                try:
                    image_response = requests.get(
                        saved_url,
                        timeout=15
                    )
                    image_response.raise_for_status()

                    saved_image = Image.open(
                        BytesIO(image_response.content)
                    ).convert("RGB")

                    image_items.append(
                        (item, saved_image)
                    )

                except Exception:
                    continue

        if not image_items:

            st.info(
                "No readable photos found in Supabase Storage. "
                "Please report a found item with a photo first."
            )

        else:

            with st.spinner("AI is comparing the images..."):

                ranked = find_similar_items(
                    query_image,
                    image_items
                )

            st.subheader("🤖 AI Image Matches")

            for (item, saved_image), score in ranked:

                with st.container(border=True):

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.image(saved_image, width=220)

                    with col2:
                        st.write(f"**Item:** {item.get('item_name', '')}")
                        st.write(f"**Description:** {item.get('description', '')}")
                        st.write(f"**Location:** {item.get('location', '')}")
                        st.write(
                            f"**Image similarity score:** "
                            f"{score * 100:.2f}%"
                        )

    except Exception as error:
        st.error(f"Unable to process the image: {error}")

st.divider()


# --------------------------------------------------
# 8. Text Search
# --------------------------------------------------


st.header("🔍 Search by Item Name or Description")

search_term = st.text_input(
    "Enter an item name, description, or location",
    key="text_search"
)

if search_term.strip():

    try:
        term = search_term.strip()

        response = (
            supabase.table("found_items")
            .select("*")
            .or_(
                f"item_name.ilike.%{term}%,"
                f"description.ilike.%{term}%,"
                f"location.ilike.%{term}%"
            )
            .order("id", desc=True)
            .execute()
        )

        search_results = response.data

        if search_results:

            for item in search_results:

                with st.container(border=True):

                    st.write(
                        f"**Item:** {item.get('item_name', '')}"
                    )

                    st.write(
                        f"**Description:** {item.get('description', '')}"
                    )

                    st.write(
                        f"**Location:** {item.get('location', '')}"
                    )

                    image_path = item.get("image_path")

                    if image_path:
                        st.image(image_path, width=220)

        else:
            st.info(
                "No matching items found. Try another search."
            )

    except Exception as error:
        st.error(f"Search failed: {error}")

st.divider()
# --------------------------------------------------
# 9. Display All Found Items
# --------------------------------------------------

st.header("📋 All Found Items")

response = (
    supabase.table("found_items")
    .select("*")
    .order("id", desc=True)
    .execute()
)

all_items = response.data

if all_items:

    st.write(
        f"Total found items reported: {len(all_items)}"
    )

    for item in all_items:

        with st.container(border=True):

            col1, col2 = st.columns([1, 2])

            with col1:

                image_path = item.get("image_path")

                if image_path:
                    st.image(image_path, width=220)
                else:
                    st.caption("No photo available")

            with col2:

                st.write(f"**Item:** {item.get('item_name', '')}")
                st.write(f"**Description:** {item.get('description', '')}")
                st.write(f"**Location:** {item.get('location', '')}")

else:

    st.info("No found items have been saved yet.")

# --------------------------------------------------
# 10. Close Database Connection
# --------------------------------------------------

connection.close()

