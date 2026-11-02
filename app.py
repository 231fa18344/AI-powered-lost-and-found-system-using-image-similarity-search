
import os
import csv
import io
import uuid
import sqlite3
import datetime
from collections import Counter

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
# PREMIUM DARK BLUE UI DESIGN
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426 0%, #0b1d35 55%, #102747 100%);
    color: #f1f5f9;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

h1, h2, h3 {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

p, label, .stMarkdown {
    color: #dbeafe;
}

div[data-testid="stMetric"] {
    background: #112944;
    border: 1px solid #245080;
    padding: 20px;
    border-radius: 14px;
}

.stButton > button {
    background: linear-gradient(90deg, #1677ff, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #3b82f6;
    color: white;
    border: 1px solid #60a5fa;
}

input, textarea {
    background-color: #102944 !important;
    color: white !important;
    border-radius: 8px !important;
}

div[data-testid="stFileUploader"] {
    background: #102944;
    border: 1px dashed #3b82f6;
    border-radius: 12px;
    padding: 15px;
}

section[data-testid="stSidebar"] {
    background: #08182b;
}

hr {
    border-color: #244568;
}
</style>
""", unsafe_allow_html=True)



# --------------------------------------------------
# 2. Directories and Database
# --------------------------------------------------

os.makedirs("item_images", exist_ok=True)

# Connect to Supabase (Public client)
supabase = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)


# --------------------------------------------------
# SEPARATE ADMIN CLIENT
# --------------------------------------------------

admin_supabase = None

# Restore admin login session after Streamlit rerun
if (
    st.session_state.get("admin_access_token")
    and st.session_state.get("admin_refresh_token")
):
    try:
        admin_supabase = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_KEY"]
        )

        admin_supabase.auth.set_session(
            st.session_state.admin_access_token,
            st.session_state.admin_refresh_token
        )

    except Exception:
        admin_supabase = None

        st.session_state.admin_access_token = None
        st.session_state.admin_refresh_token = None
        st.session_state.admin_logged_in = False


# --------------------------------------------------
# SEPARATE STUDENT CLIENT
# --------------------------------------------------

student_supabase = None

# Restore student login session after Streamlit rerun
if (
    st.session_state.get("student_access_token")
    and st.session_state.get("student_refresh_token")
):
    try:
        student_supabase = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_KEY"]
        )

        student_supabase.auth.set_session(
            st.session_state.student_access_token,
            st.session_state.student_refresh_token
        )

    except Exception:
        student_supabase = None

        st.session_state.student_access_token = None
        st.session_state.student_refresh_token = None
        st.session_state.student_logged_in = False
        st.session_state.student_user_id = None
        st.session_state.student_email = None


# --------------------------------------------------
# INITIALIZE STUDENT SESSION VARIABLES
# --------------------------------------------------

if "student_logged_in" not in st.session_state:
    st.session_state.student_logged_in = False

if "student_access_token" not in st.session_state:
    st.session_state.student_access_token = None

if "student_refresh_token" not in st.session_state:
    st.session_state.student_refresh_token = None

if "student_user_id" not in st.session_state:
    st.session_state.student_user_id = None

if "student_email" not in st.session_state:
    st.session_state.student_email = None


# --------------------------------------------------
# SQLITE DATABASE CONNECTION
# --------------------------------------------------

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


# --------------------------------------------------
# ADD IMAGE PATH COLUMN FOR EXISTING DATABASES
# --------------------------------------------------

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
# 5. TOP NAVIGATION AND WEBSITE PAGES
# --------------------------------------------------

# PREMIUM MOBILE-FRIENDLY TOP NAVIGATION

st.markdown("""
<style>

/* -----------------------------------------
   NAVIGATION MAIN PANEL
----------------------------------------- */

div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    background: linear-gradient(145deg, #101f36, #142b48);
    padding: 15px;
    border-radius: 18px;
    border: 1px solid #344966;
    width: 100%;
    box-sizing: border-box;
}

/* Hide radio circles */

div[data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}

div[data-testid="stRadio"] label > div:first-child {
    display: none !important;
}

/* -----------------------------------------
   NAVIGATION CARDS
----------------------------------------- */

div[data-testid="stRadio"] label {
    display: flex !important;
    justify-content: center;
    align-items: center;
    background: #1b2d49;
    border: 1px solid #344966;
    border-radius: 13px;
    padding: 14px 8px;
    min-height: 54px;
    width: 100%;
    box-sizing: border-box;
    margin: 0 !important;
    cursor: pointer;
    transition: all 0.2s ease;
}

/* Navigation text */

div[data-testid="stRadio"] label p {
    color: #f1f5f9 !important;
    font-size: 14px;
    font-weight: 600;
    text-align: center;
    margin: 0;
    line-height: 1.4;
}

/* Hover effect */

div[data-testid="stRadio"] label:hover {
    background: #293e5e;
    border-color: #5b789f;
    transform: translateY(-2px);
}

/* Active navigation card */

div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, #397d72, #28695f);
    border: 1px solid #69c5ad;
    box-shadow: 0 0 12px rgba(57, 125, 114, 0.3);
}

/* Active text */

div[data-testid="stRadio"] label:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 700;
}

/* Admin full width */

div[data-testid="stRadio"] label:last-child {
    grid-column: 1 / -1;
}

/* -----------------------------------------
   MOBILE RESPONSIVE DESIGN
----------------------------------------- */

@media screen and (max-width: 768px) {

    div[data-testid="stRadio"] > div[role="radiogroup"] {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 9px;
        padding: 11px;
    }

    div[data-testid="stRadio"] label {
        min-height: 50px;
        padding: 11px 5px;
        border-radius: 12px;
    }

    div[data-testid="stRadio"] label p {
        font-size: 12px;
        line-height: 1.4;
    }

    /* Mobile headings */

    h1 {
        font-size: 28px !important;
    }

    h2 {
        font-size: 23px !important;
    }

    h3 {
        font-size: 19px !important;
    }

    /* Mobile page spacing */

    section.main > div {
        padding-left: 12px;
        padding-right: 12px;
    }

    /* Responsive images */

    div[data-testid="stImage"] img {
        max-width: 100%;
        height: auto;
        object-fit: contain;
    }

    /* Mobile buttons */

    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        width: 100%;
        min-height: 44px;
        border-radius: 10px;
    }

    /* Mobile inputs */

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        width: 100%;
        box-sizing: border-box;
    }

}

/* Small mobile screens */

@media screen and (max-width: 400px) {

    div[data-testid="stRadio"] label {
        padding: 10px 3px;
        min-height: 46px;
    }

    div[data-testid="stRadio"] label p {
        font-size: 11px;
    }

}

/* -----------------------------------------
   DESKTOP RESPONSIVE DESIGN
----------------------------------------- */

@media screen and (min-width: 769px) {

    div[data-testid="stRadio"] > div[role="radiogroup"] {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }

}

/* -----------------------------------------
   GENERAL IMAGE AND CONTENT STYLING
----------------------------------------- */

div[data-testid="stImage"] {
    max-width: 100%;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TOP NAVIGATION
# --------------------------------------------------

selected_page = st.radio(
    "Navigation",
    [
        "🏠 Home",
        "📦 Report Item",
        "🔍 Search Items",
        "🤖 AI Image Search",
        "📋 All Found Items",
        "📩 Recovery Requests",
        "📤 Share",
        "👤 Student Login",
        "🔐 Admin"
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation"
)

st.divider()


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if selected_page == "🏠 Home":

    st.title("🔎 FindIt AI")

    st.subheader(
        "Your Smart Campus Lost-and-Found Assistant"
    )

    st.write(
        "Lost something on campus? Don't worry! "
        "FindIt AI helps you find your lost belongings "
        "quickly and easily using Artificial Intelligence."
    )

    st.divider()

    # ABOUT FINDIT AI

    st.header("👋 Welcome to FindIt AI")

    st.write(
        "FindIt AI is an intelligent lost-and-found "
        "platform designed especially for students."
    )

    st.write(
        "Students can report items they have found "
        "and search for belongings they have lost. "
        "The system allows users to search using item "
        "names, descriptions, or uploaded images."
    )

    st.write(
        "Using AI-powered image similarity technology, "
        "FindIt AI helps identify visually similar items "
        "even when their descriptions are different."
    )

    st.info(
        "🎯 Our Mission: To make finding lost belongings "
        "on campus simple, smart, and convenient."
    )

    st.divider()

    # HOW IT WORKS

    st.header("⚡ How Does FindIt AI Work?")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 1️⃣ Report")

        st.write(
            "Found an item? Upload its photo and provide "
            "its name, description, and location."
        )

    with col2:

        st.markdown("### 2️⃣ Search")

        st.write(
            "Lost an item? Search using its name, "
            "description, or upload a photo to find "
            "similar items."
        )

    with col3:

        st.markdown("### 3️⃣ Recover")

        st.write(
            "Identify a possible match and submit "
            "a recovery request to help get your "
            "belonging back."
        )

    st.divider()

    # MAIN FEATURES

    st.header("✨ Explore Our Features")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📦 Report Found Items")

        st.write(
            "Report items found on campus by adding "
            "their photos and details."
        )

        st.markdown("### 🔍 Text Search")

        st.write(
            "Search for lost belongings using item "
            "names and descriptions."
        )

        st.markdown("### 📋 All Found Items")

        st.write(
            "View the list of items reported by students."
        )

    with col2:

        st.markdown("### 🤖 AI Image Search")

        st.write(
            "Upload an image and find visually similar "
            "items using AI technology."
        )

        st.markdown("### 📩 Recovery Requests")

        st.write(
            "Submit a request when you identify an item "
            "that may belong to you."
        )

        st.markdown("### 👤 Student Account")

        st.write(
            "Students can create an account, manage "
            "their profile, and track recovery requests."
        )

        st.markdown("### 🔐 Admin Dashboard")

        st.write(
            "Authorized administrators can review "
            "recovery requests and update their status."
        )

    st.divider()

    # TYPES OF ITEMS

    st.header("🎒 What Can You Find?")

    st.write(
        "You can report and search for different "
        "personal belongings, including:"
    )

    item_col1, item_col2, item_col3 = st.columns(3)

    with item_col1:
        st.write("🪪 ID Cards")
        st.write("👛 Wallets")

    with item_col2:
        st.write("🔑 Keys")
        st.write("🎒 Bags")

    with item_col3:
        st.write("🎧 Earphones")
        st.write("📱 Other belongings")

    st.divider()

    # FOOTER

    st.success(
        "💙 FindIt AI — Helping Lost Items Find Their Way Home!"
    )

    st.caption(
        "An AI-powered campus lost-and-found system."
    )


# --------------------------------------------------
# SHARE PAGE
# --------------------------------------------------

elif selected_page == "📤 Share":

    st.title("🔎 FindIt AI")

    st.header("📱 Share FindIt AI")

    website_url = (
        "https://ai-powered-lost-and-found-system-using-image-similarity-search.streamlit.app/"
    )

    qr = qrcode.QRCode(box_size=10, border=4)

    qr.add_data(website_url)
    qr.make(fit=True)

    qr_image = qr.make_image(
        fill_color="black",
        back_color="white"
    )

    qr_buffer = BytesIO()

    qr_image.save(qr_buffer, format="PNG")

    qr_bytes = qr_buffer.getvalue()

    st.image(
        qr_bytes,
        caption="Scan to open FindIt AI",
        width=250
    )

    st.download_button(
        label="⬇️ Download QR Code",
        data=qr_bytes,
        file_name="FindItAI_QR_Code.png",
        mime="image/png"
    )

# --------------------------------------------------
# 6. REPORT A FOUND ITEM
# --------------------------------------------------

elif selected_page == "📦 Report Item":

    st.title("🔎 FindIt AI")

    st.header("📦 Report a Found Item")

    st.write(
        "Found something on campus? "
        "Report it here to help its owner find it."
    )

    st.divider()

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
            "📦 Submit Found Item"
        )

    if submit_item:

        if (
            item_name.strip()
            and description.strip()
            and location.strip()
        ):

            try:

                image_path = None

                # Upload image to Supabase Storage
                if uploaded_image is not None:

                    extension = os.path.splitext(
                        uploaded_image.name
                    )[1].lower()

                    image_filename = (
                        f"{uuid.uuid4().hex}{extension}"
                    )

                    image_bytes = uploaded_image.getvalue()

                    supabase.storage.from_(
                        "item-images"
                    ).upload(
                        path=image_filename,
                        file=image_bytes,
                        file_options={
                            "content-type": uploaded_image.type,
                            "upsert": "false"
                        }
                    )

                    image_path = supabase.storage.from_(
                        "item-images"
                    ).get_public_url(image_filename)

                # Save item details in Supabase
                supabase.table("found_items").insert({
                    "item_name": item_name.strip(),
                    "description": description.strip(),
                    "location": location.strip(),
                    "image_path": image_path
                }).execute()

                st.success(
                    "✅ Found item submitted successfully!"
                )

                st.balloons()

            except Exception as e:

                st.error(
                    f"Unable to submit the item: {e}"
                )

        else:

            st.error(
                "Please fill in all required fields."
            )



# --------------------------------------------------
# 7. AI IMAGE SEARCH
# --------------------------------------------------

elif selected_page == "🤖 AI Image Search":

    st.title("🔎 FindIt AI")
    st.header("🤖 Search Found Items Using AI")

    st.write(
        "Upload an image or take a photo to find "
        "visually similar items."
    )

    st.divider()

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

            database_items = response.data or []

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

                        image_items.append((item, saved_image))

                    except Exception:
                        continue

            if not image_items:

                st.info(
                    "No readable photos found in Supabase Storage. "
                    "Please report a found item with a photo first."
                )

            else:

                with st.spinner("🤖 AI is comparing the images..."):

                    ranked = find_similar_items(
                        query_image,
                        image_items
                    )

                st.subheader("🤖 AI Image Matches")

                if not ranked:

                    st.info("No similar items found.")

                else:

                    for (item, saved_image), score in ranked:

                        with st.container(border=True):

                            col1, col2 = st.columns([1, 2])

                            with col1:
                                st.image(saved_image, width=220)

                            with col2:

                                st.write(
                                    f"**Item:** "
                                    f"{item.get('item_name', '')}"
                                )

                                st.write(
                                    f"**Description:** "
                                    f"{item.get('description', '')}"
                                )

                                st.write(
                                    f"**Location:** "
                                    f"{item.get('location', '')}"
                                )

                                st.write(
                                    f"**Image similarity score:** "
                                    f"{score * 100:.2f}%"
                                )

                            # ----------------------------------
                            # DIRECT RECOVERY REQUEST
                            # ----------------------------------

                            item_id = item.get("id")

                            if st.button(
                                "🙋 This is My Item",
                                key=f"ai_recover_{item_id}",
                                use_container_width=True
                            ):

                                if not st.session_state.get(
                                    "student_logged_in", False
                                ):

                                    st.warning(
                                        "Please login to your Student "
                                        "Account before requesting recovery."
                                    )

                                elif (
                                    not st.session_state.get("student_user_id")
                                    or student_supabase is None
                                ):

                                    st.error(
                                        "Your student session expired. "
                                        "Please login again."
                                    )

                                else:

                                    try:

                                        student_id = st.session_state.get(
                                            "student_user_id"
                                        )

                                        # FETCH STUDENT PROFILE

                                        profile_response = (
                                            student_supabase
                                            .table("student_profiles")
                                            .select(
                                                "full_name, email, phone"
                                            )
                                            .eq("id", student_id)
                                            .maybe_single()
                                            .execute()
                                        )

                                        student_profile = (
                                            profile_response.data or {}
                                        )

                                        student_name = (
                                            student_profile.get(
                                                "full_name", ""
                                            )
                                        )

                                        student_email = (
                                            student_profile.get(
                                                "email",
                                                st.session_state.get(
                                                    "student_email", ""
                                                )
                                            )
                                        )

                                        student_phone = (
                                            student_profile.get(
                                                "phone", ""
                                            )
                                        )

                                        if not student_name.strip():

                                            st.error(
                                                "Please complete your "
                                                "student profile first."
                                            )

                                        elif not student_email.strip():

                                            st.error(
                                                "Your email is missing "
                                                "from your student profile."
                                            )

                                        else:

                                            student_contact = (
                                                f"Email: {student_email}"
                                            )

                                            if student_phone.strip():

                                                student_contact += (
                                                    f" | Phone: "
                                                    f"{student_phone}"
                                                )

                                            # CHECK EXISTING REQUEST

                                            existing_request = (
                                                student_supabase
                                                .table("recovery_requests")
                                                .select("id")
                                                .eq("item_id", item_id)
                                                .eq("user_id", student_id)
                                                .limit(1)
                                                .execute()
                                            )

                                            if existing_request.data:

                                                st.warning(
                                                    "You have already "
                                                    "requested this item."
                                                )

                                            else:

                                                # INSERT RECOVERY REQUEST

                                                student_supabase.table(
                                                    "recovery_requests"
                                                ).insert(
                                                    {
                                                        "item_id": item_id,
                                                        "user_id": student_id,
                                                        "requester_name": (
                                                            student_name.strip()
                                                        ),
                                                        "requester_contact": (
                                                            student_contact
                                                        ),
                                                        "message": (
                                                            "Student identified "
                                                            "this item as their "
                                                            "lost belonging."
                                                        )
                                                    },
                                                    returning="minimal"
                                                ).execute()

                                                st.success(
                                                    "🎉 Recovery request "
                                                    "submitted successfully!"
                                                )

                                                st.info(
                                                    "You can track your request "
                                                    "in Student Login → "
                                                    "My Recovery Requests."
                                                )

                                                st.rerun()

                                    except Exception as submit_error:

                                        st.error(
                                            "Unable to submit recovery request."
                                        )

                                        st.code(str(submit_error))

        except Exception as error:

            st.error(
                f"Unable to process the image: {error}"
            )

    else:

        st.info(
            "Please upload an image or take a photo "
            "to start AI image search."
        )

    st.divider()


# --------------------------------------------------
# 8. ADVANCED TEXT SEARCH, AUTOCOMPLETE
#    AND SMART LOCATION MATCHING
# --------------------------------------------------

elif selected_page == "🔍 Search Items":

    import re
    import datetime
    from difflib import SequenceMatcher

    st.title("🔎 FindIt AI")
    st.header("🔍 Advanced Item Search")

    st.write(
        "Search for found items using item name, description, "
        "or location. Use smart filters to find matching items."
    )

    st.divider()

    try:

        # LOAD ALL FOUND ITEMS

        response = (
            supabase.table("found_items")
            .select("*")
            .order("id", desc=True)
            .execute()
        )

        all_search_items = response.data or []

        if not all_search_items:

            st.info("No found items are available yet.")

        else:

            # CATEGORY DETECTION

            def get_item_category(item):

                item_text = (
                    str(item.get("item_name", "")) + " " +
                    str(item.get("description", ""))
                ).lower()

                if any(word in item_text for word in
                       ["wallet", "purse"]):
                    return "Wallet"

                elif any(word in item_text for word in
                         ["id card", "identity card", "college card"]):
                    return "ID Card"

                elif any(word in item_text for word in
                         ["key", "keys"]):
                    return "Keys"

                elif any(word in item_text for word in
                         ["bag", "backpack", "school bag"]):
                    return "Bags"

                elif any(word in item_text for word in
                         ["earphone", "headphone", "airpod"]):
                    return "Earphones"

                elif any(word in item_text for word in
                         ["mobile", "phone", "smartphone"]):
                    return "Mobile Phones"

                else:
                    return "Other"

            # LOCATION ALIASES

            LOCATION_ALIASES = [
                {"vu", "vignan", "vignan university"},
                {"library", "central library"},
                {"canteen", "college canteen"}
            ]

            def normalize_location(location):

                return re.sub(
                    r"[^a-z0-9 ]",
                    "",
                    str(location).lower()
                ).strip()

            # SMART LOCATION MATCHING

            def locations_match(location1, location2):

                a = normalize_location(location1)
                b = normalize_location(location2)

                if not a or not b:
                    return False

                if a == b:
                    return True

                for group in LOCATION_ALIASES:

                    if a in group and b in group:
                        return True

                if a in b or b in a:
                    return True

                similarity = SequenceMatcher(
                    None, a, b
                ).ratio()

                return similarity >= 0.75

            # AUTOCOMPLETE CALLBACKS

            def choose_category(value):
                st.session_state["category_autocomplete"] = value

            def choose_location(value):
                st.session_state["location_autocomplete"] = value

            # MAIN TEXT SEARCH

            st.subheader("🔎 Search Items")

            search_term = st.text_input(
                "Enter item name, description, or location",
                placeholder="Example: Black wallet, ID card...",
                key="advanced_text_search"
            )

            # FILTER SECTION

            st.subheader("🎯 Filter Your Search")

            col1, col2 = st.columns(2)

            category_options = [
                "Wallet",
                "ID Card",
                "Keys",
                "Bags",
                "Earphones",
                "Mobile Phones",
                "Other"
            ]

            # CATEGORY AUTOCOMPLETE

            with col1:

                category_query = st.text_input(
                    "📦 Search Category",
                    placeholder="Type category name...",
                    key="category_autocomplete"
                )

                category_matches = [
                    category
                    for category in category_options
                    if category_query.strip().lower()
                    in category.lower()
                ]

                if category_query.strip():

                    if category_matches:

                        st.caption("Category suggestions:")

                        for category in category_matches:

                            st.button(
                                f"📦 {category}",
                                key=f"category_suggestion_{category}",
                                on_click=choose_category,
                                args=(category,),
                                use_container_width=True
                            )

                    else:
                        st.caption("No matching categories found.")

            # GET AVAILABLE LOCATIONS

            locations = sorted(
                set(
                    str(item.get("location", "")).strip()
                    for item in all_search_items
                    if item.get("location")
                ),
                key=str.lower
            )

            # LOCATION AUTOCOMPLETE

            with col2:

                location_query = st.text_input(
                    "📍 Search Location",
                    placeholder="Type location name...",
                    key="location_autocomplete"
                )

                location_matches = [
                    location
                    for location in locations
                    if location_query.strip().lower()
                    in location.lower()
                ]

                if location_query.strip():

                    for group in LOCATION_ALIASES:

                        for alias in group:

                            if (
                                location_query.strip().lower()
                                in alias.lower()
                                and alias not in location_matches
                            ):
                                location_matches.append(alias)

                if location_query.strip():

                    if location_matches:

                        st.caption("Location suggestions:")

                        for index, location in enumerate(location_matches):

                            st.button(
                                f"📍 {location}",
                                key=f"location_suggestion_{index}_{location}",
                                on_click=choose_location,
                                args=(location,),
                                use_container_width=True
                            )

                    else:
                        st.caption("No matching locations found.")

            # SELECTED FILTER VALUES

            selected_category = (
                category_query.strip()
                if category_query.strip() in category_options
                else "All Categories"
            )

            selected_location = location_query.strip()

            # DATE FILTER

            date_filter_available = any(
                item.get("created_at")
                for item in all_search_items
            )

            selected_start_date = None
            selected_end_date = None

            if date_filter_available:

                st.subheader("📅 Date Filter")

                date_col1, date_col2 = st.columns(2)

                with date_col1:

                    selected_start_date = st.date_input(
                        "From Date",
                        value=None,
                        key="search_start_date"
                    )

                with date_col2:

                    selected_end_date = st.date_input(
                        "To Date",
                        value=None,
                        key="search_end_date"
                    )

            st.divider()

            # APPLY ALL FILTERS

            filtered_items = []

            for item in all_search_items:

                if search_term.strip():

                    term = search_term.strip().lower()

                    searchable_text = (
                        str(item.get("item_name", "")) + " " +
                        str(item.get("description", "")) + " " +
                        str(item.get("location", ""))
                    ).lower()

                    if term not in searchable_text:
                        continue

                item_category = get_item_category(item)

                if (
                    selected_category != "All Categories"
                    and item_category != selected_category
                ):
                    continue

                if selected_location:

                    item_location = item.get("location", "")

                    if not locations_match(
                        item_location,
                        selected_location
                    ):
                        continue

                if date_filter_available and (
                    selected_start_date or selected_end_date
                ):

                    created_value = item.get("created_at")

                    if not created_value:
                        continue

                    try:

                        created_date = datetime.datetime.fromisoformat(
                            str(created_value).replace("Z", "+00:00")
                        ).date()

                        if (
                            selected_start_date
                            and created_date < selected_start_date
                        ):
                            continue

                        if (
                            selected_end_date
                            and created_date > selected_end_date
                        ):
                            continue

                    except (ValueError, TypeError):
                        continue

                filtered_items.append(item)

            # DISPLAY SEARCH RESULTS

            st.subheader("🔎 Search Results")

            if filtered_items:

                st.success(
                    f"{len(filtered_items)} matching item(s) found!"
                )

                for item in filtered_items:

                    with st.container(border=True):

                        col1, col2 = st.columns([1, 2])

                        with col1:

                            image_path = item.get("image_path")

                            if image_path:

                                st.image(
                                    image_path,
                                    width=220
                                )

                            else:

                                st.info("No image available")

                        with col2:

                            st.write(
                                f"**Item:** "
                                f"{item.get('item_name', '')}"
                            )

                            st.write(
                                f"**Category:** "
                                f"{get_item_category(item)}"
                            )

                            st.write(
                                f"**Description:** "
                                f"{item.get('description', '')}"
                            )

                            st.write(
                                f"**Location:** "
                                f"{item.get('location', '')}"
                            )

                            if item.get("created_at"):

                                st.write(
                                    f"**Date Added:** "
                                    f"{str(item.get('created_at'))[:10]}"
                                )

                        # ----------------------------------
                        # DIRECT RECOVERY REQUEST
                        # ----------------------------------

                        item_id = item.get("id")

                        if st.button(
                            "🙋 This is My Item",
                            key=f"text_recover_{item_id}",
                            use_container_width=True
                        ):

                            if not st.session_state.get(
                                "student_logged_in", False
                            ):

                                st.warning(
                                    "Please login to your Student "
                                    "Account before requesting recovery."
                                )

                            elif (
                                not st.session_state.get("student_user_id")
                                or student_supabase is None
                            ):

                                st.error(
                                    "Your student session expired. "
                                    "Please login again."
                                )

                            else:

                                try:

                                    student_id = st.session_state.get(
                                        "student_user_id"
                                    )

                                    # FETCH STUDENT PROFILE

                                    profile_response = (
                                        student_supabase
                                        .table("student_profiles")
                                        .select(
                                            "full_name, email, phone"
                                        )
                                        .eq("id", student_id)
                                        .maybe_single()
                                        .execute()
                                    )

                                    student_profile = (
                                        profile_response.data or {}
                                    )

                                    student_name = (
                                        student_profile.get(
                                            "full_name", ""
                                        )
                                    )

                                    student_email = (
                                        student_profile.get(
                                            "email",
                                            st.session_state.get(
                                                "student_email", ""
                                            )
                                        )
                                    )

                                    student_phone = (
                                        student_profile.get(
                                            "phone", ""
                                        )
                                    )

                                    if not student_name.strip():

                                        st.error(
                                            "Please complete your "
                                            "student profile first."
                                        )

                                    elif not student_email.strip():

                                        st.error(
                                            "Your email is missing "
                                            "from your student profile."
                                        )

                                    else:

                                        student_contact = (
                                            f"Email: {student_email}"
                                        )

                                        if student_phone.strip():

                                            student_contact += (
                                                f" | Phone: "
                                                f"{student_phone}"
                                            )

                                        # CHECK EXISTING REQUEST

                                        existing_request = (
                                            student_supabase
                                            .table("recovery_requests")
                                            .select("id")
                                            .eq("item_id", item_id)
                                            .eq("user_id", student_id)
                                            .limit(1)
                                            .execute()
                                        )

                                        if existing_request.data:

                                            st.warning(
                                                "You have already "
                                                "requested this item."
                                            )

                                        else:

                                            # INSERT RECOVERY REQUEST

                                            student_supabase.table(
                                                "recovery_requests"
                                            ).insert(
                                                {
                                                    "item_id": item_id,
                                                    "user_id": student_id,
                                                    "requester_name": (
                                                        student_name.strip()
                                                    ),
                                                    "requester_contact": (
                                                        student_contact
                                                    ),
                                                    "message": (
                                                        "Student identified "
                                                        "this item as their "
                                                        "lost belonging."
                                                    )
                                                },
                                                returning="minimal"
                                            ).execute()

                                            st.success(
                                                "🎉 Recovery request "
                                                "submitted successfully!"
                                            )

                                            st.info(
                                                "You can track your request "
                                                "in Student Login → "
                                                "My Recovery Requests."
                                            )

                                            st.rerun()

                                except Exception as submit_error:

                                    st.error(
                                        "Unable to submit recovery request."
                                    )

                                    st.code(str(submit_error))

            else:

                st.warning(
                    "No matching items found. "
                    "Try changing your search or filters."
                )

    except Exception as error:

        st.error("Unable to search found items.")
        st.code(str(error))

    st.divider()
# --------------------------------------------------
# 9. DISPLAY ALL FOUND ITEMS
# --------------------------------------------------

elif selected_page == "📋 All Found Items":

    st.title("🔎 FindIt AI")

    st.header("📋 All Found Items")

    st.write(
        "View all the items reported by students."
    )

    st.divider()

    try:

        response = (
            supabase.table("found_items")
            .select("*")
            .order("id", desc=True)
            .execute()
        )

        all_items = response.data

        if all_items:

            st.success(
                f"Total found items reported: {len(all_items)}"
            )

            for item in all_items:

                with st.container(border=True):

                    col1, col2 = st.columns([1, 2])

                    with col1:

                        image_path = item.get("image_path")

                        if image_path:

                            st.image(
                                image_path,
                                width=220
                            )

                        else:

                            st.caption(
                                "No photo available"
                            )

                    with col2:

                        st.write(
                            f"**Item:** "
                            f"{item.get('item_name', '')}"
                        )

                        st.write(
                            f"**Description:** "
                            f"{item.get('description', '')}"
                        )

                        st.write(
                            f"**Location:** "
                            f"{item.get('location', '')}"
                        )

        else:

            st.info(
                "No found items have been saved yet."
            )

    except Exception as error:

        st.error(
            f"Unable to load found items: {error}"
        )

    st.divider()

# --------------------------------------------------
# 10. Safe Item Recovery Request
# --------------------------------------------------

elif selected_page == "📩 Recovery Requests":

    st.title("🔎 FindIt AI")
    st.header("🤝 Request Item Recovery")

    try:
        # Load found items from Supabase
        response = (
            supabase.table("found_items")
            .select("id, item_name, description, location")
            .order("id", desc=True)
            .execute()
        )

        recovery_items = response.data or []

        if not recovery_items:
            st.info("No found items are available to request.")

        else:
            # Prepare item choices
            item_options = {
                (
                    f"{item.get('item_name') or 'Unknown item'} - "
                    f"{item.get('location') or 'Location not specified'} "
                    f"(ID: {item['id']})"
                ): item["id"]
                for item in recovery_items
            }

            selected_item = st.selectbox(
                "Select the item you want to recover",
                options=list(item_options.keys()),
                key="recovery_item_select"
            )

            st.write(
                "Enter your details to request recovery of this item."
            )

            # Check whether student is logged in
            student_is_logged_in = st.session_state.get(
                "student_logged_in", False
            )

            if student_is_logged_in:
                st.success(
                    "Logged in as: "
                    + st.session_state.get("student_email", "")
                )
            else:
                st.info(
                    "You can submit a request without logging in. "
                    "Login to your student account to track your requests."
                )

            with st.form("recovery_request_form"):

                requester_name = st.text_input(
                    "Your name",
                    max_chars=100
                )

                requester_contact = st.text_input(
                    "Your contact (email or phone)",
                    max_chars=200
                )

                message = st.text_area(
                    "Explain why you think this item belongs to you",
                    max_chars=1000,
                    placeholder=(
                        "Example: I lost my wallet near the classroom. "
                        "I can describe its contents to verify ownership."
                    )
                )

                st.caption(
                    "Do not include passwords, OTPs, or sensitive ID numbers."
                )

                submitted = st.form_submit_button(
                    "📩 Submit Recovery Request"
                )

            if submitted:

                if not requester_name.strip():
                    st.warning("Please enter your name.")

                elif not requester_contact.strip():
                    st.warning("Please enter your contact details.")

                elif not message.strip():
                    st.warning("Please explain your recovery request.")

                else:
                    try:
                        # Attach logged-in student's ID
                        student_user_id = (
                            st.session_state.get("student_user_id")
                            if student_is_logged_in
                            else None
                        )

                        # Submit recovery request
                        supabase.table("recovery_requests").insert(
                            {
                                "item_id": item_options[selected_item],
                                "requester_name": requester_name.strip(),
                                "requester_contact": requester_contact.strip(),
                                "message": message.strip(),
                                "user_id": student_user_id
                            },
                            returning="minimal"
                        ).execute()

                        st.success(
                            "Recovery request submitted successfully!"
                        )

                        if student_user_id:
                            st.info(
                                "Your request is linked to your student "
                                "account. Check My Recovery Requests "
                                "in Student Login."
                            )
                        else:
                            st.info(
                                "Your request has been saved. "
                                "Login is required to track requests "
                                "through your student account."
                            )

                    except Exception as submit_error:
                        st.error("Unable to submit recovery request.")
                        st.code(str(submit_error))

    except Exception as error:
        st.error("Unable to load the recovery form.")
        st.code(str(error))

    st.divider()
    


# --------------------------------------------------
# 11. STUDENT LOGIN, SIGN UP, PROFILE,
#     MY RECOVERY REQUESTS AND ADMIN DASHBOARD
# -------------------------------------------------

# ==================================================
# STUDENT LOGIN AND PROFILE
# ==================================================

elif selected_page == "👤 Student Login":

    st.title("🔎 FindIt AI")
    st.header("👤 Student Login")

    if "student_logged_in" not in st.session_state:
        st.session_state.student_logged_in = False

    if "student_access_token" not in st.session_state:
        st.session_state.student_access_token = None

    if "student_refresh_token" not in st.session_state:
        st.session_state.student_refresh_token = None

    if "student_user_id" not in st.session_state:
        st.session_state.student_user_id = None

    if not st.session_state.student_logged_in:

        login_tab, signup_tab = st.tabs([
            "🔐 Login",
            "📝 Sign Up"
        ])

        # ------------------------------------------
        # STUDENT LOGIN
        # ------------------------------------------

        with login_tab:

            with st.form("student_login_form"):

                student_email = st.text_input(
                    "Email Address",
                    key="student_login_email"
                )

                student_password = st.text_input(
                    "Password",
                    type="password",
                    key="student_login_password"
                )

                login_button = st.form_submit_button(
                    "Login"
                )

            if login_button:

                if not student_email.strip() or not student_password:

                    st.warning(
                        "Please enter your email and password."
                    )

                else:

                    try:

                        login_client = create_client(
                            st.secrets["SUPABASE_URL"],
                            st.secrets["SUPABASE_KEY"]
                        )

                        login_response = (
                            login_client.auth.sign_in_with_password({
                                "email": student_email.strip(),
                                "password": student_password
                            })
                        )

                        logged_user = login_response.user
                        student_session = login_response.session

                        if logged_user and student_session:

                            st.session_state.student_access_token = (
                                student_session.access_token
                            )

                            st.session_state.student_refresh_token = (
                                student_session.refresh_token
                            )

                            st.session_state.student_user_id = str(
                                logged_user.id
                            )

                            st.session_state.student_email = (
                                logged_user.email
                            )

                            st.session_state.student_logged_in = True

                            st.success("Student login successful!")

                            st.rerun()

                        else:

                            st.error(
                                "Login failed. Please verify your account."
                            )

                    except Exception as login_error:

                        st.error("Unable to login.")

                        st.code(str(login_error))

        # ------------------------------------------
        # STUDENT SIGN UP
        # ------------------------------------------

        with signup_tab:

            with st.form("student_signup_form"):

                signup_name = st.text_input(
                    "Full Name",
                    key="student_signup_name"
                )

                signup_email = st.text_input(
                    "Email Address",
                    key="student_signup_email"
                )

                signup_phone = st.text_input(
                    "Phone Number",
                    max_chars=10,
                    key="student_signup_phone"
                )

                signup_password = st.text_input(
                    "Create Password",
                    type="password",
                    key="student_signup_password"
                )

                signup_button = st.form_submit_button(
                    "Create Account"
                )

            if signup_button:

                if not signup_name.strip():

                    st.warning("Please enter your full name.")

                elif not signup_email.strip():

                    st.warning("Please enter your email.")

                elif (
                    not signup_phone.isdigit()
                    or len(signup_phone) != 10
                ):

                    st.warning(
                        "Enter a valid 10-digit phone number."
                    )

                elif len(signup_password) < 6:

                    st.warning(
                        "Password must contain at least 6 characters."
                    )

                else:

                    try:

                        signup_client = create_client(
                            st.secrets["SUPABASE_URL"],
                            st.secrets["SUPABASE_KEY"]
                        )

                        signup_response = signup_client.auth.sign_up({
                            "email": signup_email.strip(),
                            "password": signup_password
                        })

                        new_user = signup_response.user
                        new_session = signup_response.session

                        if new_user:

                            if new_session:

                                profile_client = create_client(
                                    st.secrets["SUPABASE_URL"],
                                    st.secrets["SUPABASE_KEY"]
                                )

                                profile_client.auth.set_session(
                                    new_session.access_token,
                                    new_session.refresh_token
                                )

                                profile_client.table(
                                    "student_profiles"
                                ).upsert({
                                    "id": str(new_user.id),
                                    "full_name": signup_name.strip(),
                                    "email": signup_email.strip(),
                                    "phone": signup_phone.strip()
                                }).execute()

                                st.session_state.student_access_token = (
                                    new_session.access_token
                                )

                                st.session_state.student_refresh_token = (
                                    new_session.refresh_token
                                )

                                st.session_state.student_user_id = str(
                                    new_user.id
                                )

                                st.session_state.student_email = (
                                    signup_email.strip()
                                )

                                st.session_state.student_logged_in = True

                                st.success(
                                    "Account created successfully!"
                                )

                                st.rerun()

                            else:

                                st.success(
                                    "Account created. Please verify your "
                                    "email and then login."
                                )

                        else:

                            st.error(
                                "Unable to create your account."
                            )

                    except Exception as signup_error:

                        st.error("Sign up failed.")

                        st.code(str(signup_error))

    # ==================================================
    # LOGGED-IN STUDENT DASHBOARD
    # ==================================================

    else:

        st.success("You are logged in successfully!")

        student_supabase = None

        try:

            student_supabase = create_client(
                st.secrets["SUPABASE_URL"],
                st.secrets["SUPABASE_KEY"]
            )

            student_supabase.auth.set_session(
                st.session_state.student_access_token,
                st.session_state.student_refresh_token
            )

        except Exception as session_error:

            st.error(
                "Your student session expired. Please login again."
            )

            st.code(str(session_error))

            st.session_state.student_logged_in = False
            st.session_state.student_access_token = None
            st.session_state.student_refresh_token = None
            st.session_state.student_user_id = None

            st.rerun()

        if student_supabase is not None:

            student_uid = st.session_state.student_user_id

            # ------------------------------------------
            # LOAD STUDENT PROFILE
            # ------------------------------------------

            try:

                profile_response = (
                    student_supabase.table("student_profiles")
                    .select("full_name, email, phone")
                    .eq("id", student_uid)
                    .maybe_single()
                    .execute()
                )

                student_profile = profile_response.data or {}

            except Exception as profile_error:

                student_profile = {}

                st.error("Unable to load student profile.")

                st.code(str(profile_error))

            st.divider()

            st.subheader("👤 My Profile")

            with st.form("student_profile_update_form"):

                profile_name = st.text_input(
                    "Full Name",
                    value=student_profile.get("full_name", "")
                )

                profile_email = st.text_input(
                    "Email Address",
                    value=student_profile.get(
                        "email",
                        st.session_state.get("student_email", "")
                    ),
                    disabled=True
                )

                profile_phone = st.text_input(
                    "Phone Number",
                    value=student_profile.get("phone", ""),
                    max_chars=10
                )

                update_profile = st.form_submit_button(
                    "Update Profile"
                )

            if update_profile:

                if not profile_name.strip():

                    st.warning("Please enter your name.")

                elif (
                    not profile_phone.isdigit()
                    or len(profile_phone) != 10
                ):

                    st.warning(
                        "Enter a valid 10-digit phone number."
                    )

                else:

                    try:

                        student_supabase.table(
                            "student_profiles"
                        ).upsert({
                            "id": student_uid,
                            "full_name": profile_name.strip(),
                            "email": profile_email,
                            "phone": profile_phone.strip()
                        }).execute()

                        st.success(
                            "Profile updated successfully!"
                        )

                        st.rerun()

                    except Exception as update_error:

                        st.error("Unable to update profile.")

                        st.code(str(update_error))

            st.divider()

            # ------------------------------------------
            # MY RECOVERY REQUESTS
            # ------------------------------------------

            st.subheader("📩 My Recovery Requests")

            try:

                my_requests_response = (
                    student_supabase.table("recovery_requests")
                    .select(
                        "id, item_id, requester_name, message, "
                        "status, created_at, "
                        "found_items(item_name, location, image_path)"
                    )
                    .eq("user_id", student_uid)
                    .order("created_at", desc=True)
                    .execute()
                )

                my_requests = my_requests_response.data or []

                if my_requests:

                    st.success(
                        f"You have submitted {len(my_requests)} request(s)."
                    )

                    for request in my_requests:

                        item_details = (
                            request.get("found_items") or {}
                        )

                        with st.container(border=True):

                            st.write(
                                f"**Item:** "
                                f"{item_details.get('item_name', 'Unknown item')}"
                            )

                            st.write(
                                f"**Location:** "
                                f"{item_details.get('location', 'Not specified')}"
                            )

                            # SHOW FOUND ITEM PHOTO
                            item_image = item_details.get("image_path")

                            if item_image:

                                st.image(
                                    item_image,
                                    caption="Found Item Photo",
                                    width=250
                                )

                            else:

                                st.info(
                                    "No photo available for this item."
                                )

                            st.write(
                                f"**Message:** "
                                f"{request.get('message', '')}"
                            )

                            st.write(
                                f"**Request Date:** "
                                f"{str(request.get('created_at', ''))[:10]}"
                            )

                            request_status = request.get(
                                "status",
                                "Pending"
                            )

                            st.write(
                                f"**Status:** {request_status}"
                            )

                else:

                    st.info(
                        "You have not submitted any recovery requests yet."
                    )

            except Exception as request_error:

                st.error(
                    "Unable to load your recovery requests."
                )

                st.code(str(request_error))

            st.divider()

            # ------------------------------------------
            # STUDENT LOGOUT
            # ------------------------------------------

            if st.button(
                "🚪 Student Logout",
                key="student_logout_button"
            ):

                try:
                    student_supabase.auth.sign_out()
                except Exception:
                    pass

                st.session_state.student_logged_in = False
                st.session_state.student_access_token = None
                st.session_state.student_refresh_token = None
                st.session_state.student_user_id = None
                st.session_state.student_email = None

                st.success("Logged out successfully!")

                st.rerun()


# ==================================================
# ADMIN LOGIN AND DASHBOARD
# ==================================================

elif selected_page == "🔐 Admin":

    ADMIN_UID = "1fa8098e-27be-46c1-9916-1e20506f9fa5"

    st.title("🔎 FindIt AI")

    st.header("🔐 Admin Login")

    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    # ----------------------------------------------
    # ADMIN LOGIN
    # ----------------------------------------------

    if not st.session_state.admin_logged_in:

        with st.form("admin_login_form"):

            admin_email = st.text_input(
                "Admin Email",
                key="admin_email_input"
            )

            admin_password = st.text_input(
                "Admin Password",
                type="password",
                key="admin_password_input"
            )

            login_button = st.form_submit_button(
                "Login"
            )

        if login_button:

            if not admin_email.strip() or not admin_password:

                st.warning(
                    "Please enter your email and password."
                )

            else:

                try:

                    login_client = create_client(
                        st.secrets["SUPABASE_URL"],
                        st.secrets["SUPABASE_KEY"]
                    )

                    login_response = (
                        login_client.auth.sign_in_with_password({
                            "email": admin_email.strip(),
                            "password": admin_password
                        })
                    )

                    logged_user = login_response.user
                    admin_session = login_response.session

                    if (
                        logged_user
                        and str(logged_user.id) == ADMIN_UID
                        and admin_session
                    ):

                        st.session_state.admin_access_token = (
                            admin_session.access_token
                        )

                        st.session_state.admin_refresh_token = (
                            admin_session.refresh_token
                        )

                        st.session_state.admin_logged_in = True

                        st.success("Admin login successful!")

                        st.rerun()

                    else:

                        login_client.auth.sign_out()

                        st.error(
                            "You are not authorized as admin."
                        )

                except Exception as login_error:

                    st.error(
                        "Login failed. Check your email and password."
                    )

                    st.code(str(login_error))

    # ----------------------------------------------
    # ADMIN DASHBOARD
    # ----------------------------------------------

    else:

        st.success("Admin logged in successfully!")

        if admin_supabase is None:

            st.error(
                "Admin session expired. Please login again."
            )

            st.session_state.admin_logged_in = False
            st.session_state.admin_access_token = None
            st.session_state.admin_refresh_token = None

            st.rerun()

        else:

            st.subheader("📋 Recovery Requests Dashboard")

            try:

                # ----------------------------------
                # LOAD RECOVERY REQUESTS
                # ----------------------------------

                requests_response = (
                    admin_supabase.table("recovery_requests")
                    .select(
                        "id, item_id, user_id, requester_name, "
                        "requester_contact, message, status, created_at"
                    )
                    .order("created_at", desc=True)
                    .execute()
                )

                recovery_requests = requests_response.data or []

                # ----------------------------------
                # LOAD FOUND ITEM DETAILS
                # ----------------------------------

                items_response = (
                    admin_supabase.table("found_items")
                    .select("*")
                    .execute()
                )

                found_items = items_response.data or []

                items_by_id = {
                    str(item.get("id")): item
                    for item in found_items
                }

                # ----------------------------------
                # LOAD STUDENT PROFILES
                # ----------------------------------

                student_ids = list(set(
                    request.get("user_id")
                    for request in recovery_requests
                    if request.get("user_id")
                ))

                profiles_by_id = {}

                if student_ids:

                    profiles_response = (
                        admin_supabase.table("student_profiles")
                        .select("id, full_name, email, phone")
                        .in_("id", student_ids)
                        .execute()
                    )

                    student_profiles = profiles_response.data or []

                    profiles_by_id = {
                        str(profile.get("id")): profile
                        for profile in student_profiles
                    }

                # ----------------------------------
                # ATTACH ITEM AND STUDENT DETAILS
                # ----------------------------------

                for request in recovery_requests:

                    request["item_details"] = items_by_id.get(
                        str(request.get("item_id")),
                        {}
                    )

                    request["student_details"] = profiles_by_id.get(
                        str(request.get("user_id")),
                        {}
                    )

                # ----------------------------------
                # REQUEST STATISTICS
                # ----------------------------------

                total_requests = len(recovery_requests)

                pending_count = sum(
                    1 for r in recovery_requests
                    if r.get("status") == "Pending"
                )

                contacted_count = sum(
                    1 for r in recovery_requests
                    if r.get("status") == "Contacted"
                )

                returned_count = sum(
                    1 for r in recovery_requests
                    if r.get("status") == "Returned"
                )

                rejected_count = sum(
                    1 for r in recovery_requests
                    if r.get("status") == "Rejected"
                )

                active_count = pending_count + contacted_count

                st.subheader("📊 Recovery Request Statistics")

                col1, col2, col3 = st.columns(3)

                col1.metric("📋 Total Requests", total_requests)
                col2.metric("⏳ Pending", pending_count)
                col3.metric("📞 Contacted", contacted_count)

                col4, col5, col6 = st.columns(3)

                col4.metric("✅ Returned", returned_count)
                col5.metric("❌ Rejected", rejected_count)
                col6.metric("🔄 Active", active_count)

                st.divider()

                # ----------------------------------
                # RECOVERY PERFORMANCE
                # ----------------------------------

                st.header("📈 Advanced Analytics")

                if total_requests:

                    recovery_rate = (
                        returned_count / total_requests
                    ) * 100

                    pending_rate = (
                        pending_count / total_requests
                    ) * 100

                    contacted_rate = (
                        contacted_count / total_requests
                    ) * 100

                else:

                    recovery_rate = 0
                    pending_rate = 0
                    contacted_rate = 0

                rate_col1, rate_col2, rate_col3 = st.columns(3)

                rate_col1.metric(
                    "Recovery Success Rate",
                    f"{recovery_rate:.1f}%"
                )

                rate_col2.metric(
                    "Pending Requests",
                    f"{pending_rate:.1f}%"
                )

                rate_col3.metric(
                    "Contacted Requests",
                    f"{contacted_rate:.1f}%"
                )

                st.progress(
                    int(round(recovery_rate))
                )

                st.divider()

                # ----------------------------------
                # STATUS CHART
                # ----------------------------------

                st.subheader("📊 Request Status Distribution")

                status_chart_data = {
                    "Status": [
                        "Pending",
                        "Contacted",
                        "Returned",
                        "Rejected"
                    ],
                    "Count": [
                        pending_count,
                        contacted_count,
                        returned_count,
                        rejected_count
                    ]
                }

                st.bar_chart(
                    status_chart_data,
                    x="Status",
                    y="Count"
                )

                st.divider()

                # ----------------------------------
                # MOST REQUESTED ITEMS
                # ----------------------------------

                st.subheader("🏆 Most Requested Items")

                item_counter = Counter()

                for request in recovery_requests:

                    item_name = request.get(
                        "item_details", {}
                    ).get("item_name", "Unknown item")

                    item_counter[item_name] += 1

                if item_counter:

                    top_items = item_counter.most_common(10)

                    item_chart_data = {
                        "Item": [
                            item[0] for item in top_items
                        ],
                        "Requests": [
                            item[1] for item in top_items
                        ]
                    }

                    st.bar_chart(
                        item_chart_data,
                        x="Item",
                        y="Requests"
                    )

                else:

                    st.info("No item request data available.")

                st.divider()

                # ----------------------------------
                # MOST REQUESTED LOCATIONS
                # ----------------------------------

                st.subheader("📍 Most Requested Locations")

                location_counter = Counter()

                for request in recovery_requests:

                    location = request.get(
                        "item_details", {}
                    ).get("location", "Not specified")

                    location_counter[location] += 1

                if location_counter:

                    top_locations = location_counter.most_common(10)

                    location_chart_data = {
                        "Location": [
                            loc[0] for loc in top_locations
                        ],
                        "Requests": [
                            loc[1] for loc in top_locations
                        ]
                    }

                    st.bar_chart(
                        location_chart_data,
                        x="Location",
                        y="Requests"
                    )

                else:

                    st.info("No location data available.")

                st.divider()

                # ----------------------------------
                # DAILY REQUEST TRENDS
                # ----------------------------------

                st.subheader("📅 Daily Recovery Trends")

                date_counts = {}

                for request in recovery_requests:

                    created_at = request.get("created_at")

                    if created_at:

                        try:

                            request_date = (
                                datetime.datetime.fromisoformat(
                                    str(created_at).replace(
                                        "Z",
                                        "+00:00"
                                    )
                                ).date()
                            )

                            date_label = request_date.isoformat()

                            date_counts[date_label] = (
                                date_counts.get(date_label, 0) + 1
                            )

                        except (ValueError, TypeError):
                            continue

                if date_counts:

                    sorted_dates = sorted(date_counts)

                    date_chart_data = {
                        "Date": sorted_dates,
                        "Requests": [
                            date_counts[date]
                            for date in sorted_dates
                        ]
                    }

                    st.line_chart(
                        date_chart_data,
                        x="Date",
                        y="Requests"
                    )

                else:

                    st.info("No daily request data available.")

                st.divider()

                # ----------------------------------
                # MONTHLY REQUEST TRENDS
                # ----------------------------------

                st.subheader("📆 Monthly Recovery Trends")

                month_counts = {}

                for request in recovery_requests:

                    created_at = request.get("created_at")

                    if created_at:

                        try:

                            request_date = (
                                datetime.datetime.fromisoformat(
                                    str(created_at).replace(
                                        "Z",
                                        "+00:00"
                                    )
                                )
                            )

                            month_label = request_date.strftime("%Y-%m")

                            month_counts[month_label] = (
                                month_counts.get(month_label, 0) + 1
                            )

                        except (ValueError, TypeError):
                            continue

                if month_counts:

                    sorted_months = sorted(month_counts)

                    monthly_chart_data = {
                        "Month": sorted_months,
                        "Requests": [
                            month_counts[month]
                            for month in sorted_months
                        ]
                    }

                    st.bar_chart(
                        monthly_chart_data,
                        x="Month",
                        y="Requests"
                    )

                else:

                    st.info("No monthly request data available.")

                st.divider()

                # ----------------------------------
                # RECENT ACTIVITY
                # ----------------------------------

                st.subheader("🕒 Recent Recovery Activity")

                if recovery_requests:

                    for request in recovery_requests[:5]:

                        item_name = request.get(
                            "item_details", {}
                        ).get("item_name", "Unknown item")

                        st.write(
                            f"📦 **{item_name}** — "
                            f"{request.get('status', 'Pending')} — "
                            f"{str(request.get('created_at', ''))[:10]}"
                        )

                else:

                    st.info("No recent recovery activity.")

                st.divider()

                # ----------------------------------
                # SEARCH AND FILTER REQUESTS
                # ----------------------------------

                st.subheader("🔎 Search Recovery Requests")

                request_search = st.text_input(
                    "Search by student name, item, email or location",
                    key="admin_request_search"
                )

                status_filter = st.selectbox(
                    "Filter by Status",
                    [
                        "All",
                        "Pending",
                        "Contacted",
                        "Returned",
                        "Rejected"
                    ],
                    key="admin_status_filter"
                )

                filtered_requests = []

                for request in recovery_requests:

                    item_details = request.get("item_details", {})
                    student_details = request.get("student_details", {})

                    searchable_text = " ".join([
                        str(request.get("requester_name", "")),
                        str(request.get("requester_contact", "")),
                        str(student_details.get("full_name", "")),
                        str(student_details.get("email", "")),
                        str(student_details.get("phone", "")),
                        str(item_details.get("item_name", "")),
                        str(item_details.get("location", ""))
                    ]).lower()

                    if (
                        request_search.strip()
                        and request_search.strip().lower()
                        not in searchable_text
                    ):
                        continue

                    if (
                        status_filter != "All"
                        and request.get("status") != status_filter
                    ):
                        continue

                    filtered_requests.append(request)

                st.write(
                    f"Showing {len(filtered_requests)} request(s)."
                )

                st.divider()

                # ----------------------------------
                # REQUEST DETAILS AND STATUS UPDATE
                # ----------------------------------

                st.subheader("📩 Recovery Request Details")

                if filtered_requests:

                    for request in filtered_requests:

                        item_details = request.get(
                            "item_details", {}
                        )

                        student_details = request.get(
                            "student_details", {}
                        )

                        request_id = request.get("id")

                        item_name = item_details.get(
                            "item_name",
                            "Unknown item"
                        )

                        student_name = student_details.get(
                            "full_name",
                            request.get("requester_name", "Unknown student")
                        )

                        with st.expander(
                            f"📦 {item_name} — {student_name} "
                            f"({request.get('status', 'Pending')})"
                        ):

                            st.markdown("### 👨‍🎓 Student Details")

                            st.write(
                                f"**Name:** {student_name}"
                            )

                            st.write(
                                f"**Email:** "
                                f"{student_details.get('email', 'Not available')}"
                            )

                            st.write(
                                f"**Phone:** "
                                f"{student_details.get('phone', 'Not available')}"
                            )

                            st.write(
                                f"**Saved Contact:** "
                                f"{request.get('requester_contact', 'Not available')}"
                            )

                            st.divider()

                            st.markdown("### 📦 Found Item Details")

                            st.write(
                                f"**Item Name:** {item_name}"
                            )

                            st.write(
                                f"**Description:** "
                                f"{item_details.get('description', 'Not available')}"
                            )

                            st.write(
                                f"**Location:** "
                                f"{item_details.get('location', 'Not available')}"
                            )

                            # SHOW FOUND ITEM PHOTO IN ADMIN DASHBOARD
                            item_image = item_details.get("image_path")

                            if item_image:

                                st.image(
                                    item_image,
                                    caption="Found Item Photo",
                                    width=250
                                )

                            else:

                                st.info(
                                    "No photo available for this item."
                                )

                            st.divider()

                            st.markdown("### 📝 Recovery Request")

                            st.write(
                                request.get(
                                    "message",
                                    "No message provided."
                                )
                            )

                            st.write(
                                f"**Submitted:** "
                                f"{str(request.get('created_at', ''))[:19]}"
                            )

                            st.write(
                                f"**Current Status:** "
                                f"{request.get('status', 'Pending')}"
                            )

                            st.divider()

                            new_status = st.selectbox(
                                "Update Request Status",
                                [
                                    "Pending",
                                    "Contacted",
                                    "Returned",
                                    "Rejected"
                                ],
                                index=[
                                    "Pending",
                                    "Contacted",
                                    "Returned",
                                    "Rejected"
                                ].index(
                                    request.get("status", "Pending")
                                    if request.get("status") in [
                                        "Pending",
                                        "Contacted",
                                        "Returned",
                                        "Rejected"
                                    ]
                                    else "Pending"
                                ),
                                key=f"admin_status_{request_id}"
                            )

                            if st.button(
                                "💾 Save Status",
                                key=f"save_status_{request_id}"
                            ):

                                try:

                                    (
                                        admin_supabase
                                        .table("recovery_requests")
                                        .update({
                                            "status": new_status
                                        })
                                        .eq("id", request_id)
                                        .execute()
                                    )

                                    st.success(
                                        "Request status updated successfully!"
                                    )

                                    st.rerun()

                                except Exception as status_error:

                                    st.error(
                                        "Unable to update request status."
                                    )

                                    st.code(str(status_error))

                else:

                    st.info(
                        "No recovery requests match your search."
                    )

                st.divider()

                # ----------------------------------
                # DOWNLOAD CSV REPORT
                # ----------------------------------

                st.subheader("📥 Download Recovery Report")

                csv_buffer = io.StringIO()

                csv_writer = csv.writer(csv_buffer)

                csv_writer.writerow([
                    "Request ID",
                    "Student Name",
                    "Student Email",
                    "Student Phone",
                    "Item Name",
                    "Item Description",
                    "Item Location",
                    "Requester Contact",
                    "Message",
                    "Status",
                    "Created At"
                ])

                for request in recovery_requests:

                    item_details = request.get(
                        "item_details", {}
                    )

                    student_details = request.get(
                        "student_details", {}
                    )

                    csv_writer.writerow([
                        request.get("id"),
                        student_details.get(
                            "full_name",
                            request.get("requester_name", "")
                        ),
                        student_details.get("email", ""),
                        student_details.get("phone", ""),
                        item_details.get("item_name", ""),
                        item_details.get("description", ""),
                        item_details.get("location", ""),
                        request.get("requester_contact", ""),
                        request.get("message", ""),
                        request.get("status", ""),
                        request.get("created_at", "")
                    ])

                st.download_button(
                    label="⬇️ Download CSV Report",
                    data=csv_buffer.getvalue(),
                    file_name="FindItAI_Recovery_Report.csv",
                    mime="text/csv"
                )

            except Exception as dashboard_error:

                st.error(
                    "Unable to load the Admin Dashboard."
                )

                st.code(str(dashboard_error))

        st.divider()

        # ------------------------------------------
        # ADMIN LOGOUT
        # ------------------------------------------

        if st.button(
            "🚪 Admin Logout",
            key="admin_logout_button"
        ):

            try:
                if admin_supabase is not None:
                    admin_supabase.auth.sign_out()
            except Exception:
                pass

            st.session_state.admin_logged_in = False
            st.session_state.admin_access_token = None
            st.session_state.admin_refresh_token = None

            st.success("Admin logged out successfully!")

            st.rerun()
# --------------------------------------------------
# 12. CLOSE DATABASE CONNECTION
# --------------------------------------------------

connection.close()
