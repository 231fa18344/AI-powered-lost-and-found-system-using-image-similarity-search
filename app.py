
import os
import csv
import io
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

# Separate Admin client
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


# SQLite Database Connection
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
# 5. TOP NAVIGATION AND WEBSITE PAGES
# --------------------------------------------------

# PREMIUM TOP NAVIGATION BAR

st.markdown("""
<style>

/* Navigation bar background */
div[data-testid="stRadio"] > div[role="radiogroup"] {
    background: #17232f;
    padding: 10px;
    border-radius: 14px;
    border: 1px solid #35434d;
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 6px;
}

/* Hide radio circles */
div[data-testid="stRadio"] label > div:first-child {
    display: none;
}

/* Navigation options */
div[data-testid="stRadio"] label {
    background: transparent;
    color: #d8e1e5 !important;
    padding: 10px 15px;
    border-radius: 9px;
    border: 1px solid transparent;
    transition: all 0.2s ease;
    cursor: pointer;
}

/* Hover effect */
div[data-testid="stRadio"] label:hover {
    background: #293b45;
    color: #ffffff !important;
}

/* Active navigation option */
div[data-testid="stRadio"] label:has(input:checked) {
    background: #397d72;
    color: #ffffff !important;
    border: 1px solid #69a99b;
    font-weight: 600;
}

/* Remove extra radio spacing */
div[data-testid="stRadio"] label p {
    font-size: 14px;
    margin: 0;
}

</style>
""", unsafe_allow_html=True)


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

            st.image(
                query_image,
                width=220
            )

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

                if (
                    saved_url
                    and saved_url.startswith("https://")
                ):

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

                with st.spinner(
                    "🤖 AI is comparing the images..."
                ):

                    ranked = find_similar_items(
                        query_image,
                        image_items
                    )

                st.subheader("🤖 AI Image Matches")

                if not ranked:

                    st.info(
                        "No similar items found."
                    )

                else:

                    for (item, saved_image), score in ranked:

                        with st.container(border=True):

                            col1, col2 = st.columns([1, 2])

                            with col1:

                                st.image(
                                    saved_image,
                                    width=220
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

                                st.write(
                                    f"**Image similarity score:** "
                                    f"{score * 100:.2f}%"
                                )

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
# 8. ADVANCED TEXT SEARCH AND FILTERS
# --------------------------------------------------

elif selected_page == "🔍 Search Items":

    st.title("🔎 FindIt AI")

    st.header("🔍 Advanced Item Search")

    st.write(
        "Search for lost belongings using item name, "
        "description, or location. Use filters to find "
        "the items more easily."
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

            # SEARCH INPUT

            st.subheader("🔎 Search Items")

            search_term = st.text_input(
                "Enter item name, description, or location",
                key="advanced_text_search"
            )

            # FILTER OPTIONS

            st.subheader("🎯 Filter Your Search")

            col1, col2 = st.columns(2)

            with col1:

                category_options = [
                    "All Categories",
                    "Wallet",
                    "ID Card",
                    "Keys",
                    "Bags",
                    "Earphones",
                    "Mobile Phones",
                    "Other"
                ]

                selected_category = st.selectbox(
                    "📦 Select Category",
                    category_options,
                    key="search_category_filter"
                )

            with col2:

                locations = sorted(
                    set(
                        str(item.get("location", "")).strip()
                        for item in all_search_items
                        if item.get("location")
                    )
                )

                location_options = [
                    "All Locations"
                ] + locations

                selected_location = st.selectbox(
                    "📍 Select Location",
                    location_options,
                    key="search_location_filter"
                )

            # DATE FILTER - ONLY IF created_at EXISTS

            date_filter_available = any(
                item.get("created_at")
                for item in all_search_items
            )

            selected_start_date = None
            selected_end_date = None

            if date_filter_available:

                import datetime

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

            else:

                st.caption(
                    "Date filtering is unavailable because "
                    "the found_items records do not contain "
                    "a created_at date."
                )

            st.divider()

            # APPLY FILTERS

            filtered_items = []

            for item in all_search_items:

                # TEXT SEARCH

                if search_term.strip():

                    term = search_term.strip().lower()

                    searchable_text = (
                        str(item.get("item_name", "")) + " " +
                        str(item.get("description", "")) + " " +
                        str(item.get("location", ""))
                    ).lower()

                    if term not in searchable_text:
                        continue

                # CATEGORY FILTER

                item_category = get_item_category(item)

                if (
                    selected_category != "All Categories"
                    and item_category != selected_category
                ):
                    continue

                # LOCATION FILTER

                if (
                    selected_location != "All Locations"
                    and str(item.get("location", "")).strip()
                    != selected_location
                ):
                    continue

                # DATE FILTER

                if date_filter_available and (
                    selected_start_date or selected_end_date
                ):

                    import datetime

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

            # DISPLAY RESULTS

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
                    "Submit Recovery Request"
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
                        # Submit request without returning row data
                        supabase.table("recovery_requests").insert(
                            {
                                "item_id": item_options[selected_item],
                                "requester_name": requester_name.strip(),
                                "requester_contact": requester_contact.strip(),
                                "message": message.strip()
                            },
                            returning="minimal"
                        ).execute()

                        st.success(
                            "Recovery request submitted successfully!"
                        )

                        st.info(
                            "Your request has been saved. "
                            "The finder has not been automatically notified."
                        )

                    except Exception as submit_error:
                        st.error("Unable to submit recovery request.")
                        st.code(str(submit_error))

    except Exception as error:
        st.error("Unable to load the recovery form.")
        st.code(str(error))

    st.divider()


# --------------------------------------------------
# 11. Admin Login and Recovery Requests Dashboard
# --------------------------------------------------

elif selected_page == "🔐 Admin":

    ADMIN_UID = "1fa8098e-27be-46c1-9916-1e20506f9fa5"

    st.title("🔎 FindIt AI")
    st.divider()
    st.header("🔐 Admin Login")

    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    if not st.session_state.admin_logged_in:

        with st.form("admin_login_form"):

            admin_email = st.text_input("Admin Email")

            admin_password = st.text_input(
                "Admin Password",
                type="password"
            )

            login_button = st.form_submit_button("Login")

        if login_button:

            if not admin_email.strip() or not admin_password:
                st.warning("Please enter email and password.")

            else:
                try:
                    login_client = create_client(
                        st.secrets["SUPABASE_URL"],
                        st.secrets["SUPABASE_KEY"]
                    )

                    login_response = login_client.auth.sign_in_with_password({
                        "email": admin_email.strip(),
                        "password": admin_password
                    })

                    logged_in_user = login_response.user
                    admin_session = login_response.session

                    if (
                        logged_in_user
                        and str(logged_in_user.id) == ADMIN_UID
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
                        st.error("You are not authorized as admin.")

                except Exception as login_error:
                    st.error("Login failed. Please check your email and password.")
                    st.code(str(login_error))

    else:

        st.success("Admin logged in successfully!")

        if admin_supabase is None:

            st.error("Admin session expired. Please login again.")

            st.session_state.admin_logged_in = False
            st.session_state.admin_access_token = None
            st.session_state.admin_refresh_token = None

            st.rerun()

        else:

            st.subheader("📋 Recovery Requests Dashboard")

            try:

                requests_response = (
                    admin_supabase.table("recovery_requests")
                    .select(
                        "id, item_id, requester_name, requester_contact, "
                        "message, status, created_at, "
                        "found_items(item_name, location)"
                    )
                    .order("created_at", desc=True)
                    .execute()
                )

                recovery_requests = requests_response.data or []

                # --------------------------------------------------
                # Recovery Request Statistics
                # --------------------------------------------------

                st.subheader("📊 Recovery Request Statistics")

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

                col1, col2, col3 = st.columns(3)

                col1.metric("📋 Total Requests", total_requests)
                col2.metric("⏳ Pending", pending_count)
                col3.metric("📞 Contacted", contacted_count)

                col4, col5 = st.columns(2)

                col4.metric("✅ Returned", returned_count)
                col5.metric("❌ Rejected", rejected_count)

                st.divider()

                # --------------------------------------------------
                # Recovery Request Analytics
                # --------------------------------------------------

                st.subheader("📈 Recovery Request Analytics")

                chart_data = {
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

                st.write("**Recovery Requests by Status**")

                st.bar_chart(
                    chart_data,
                    x="Status",
                    y="Count"
                )

                st.divider()

                # --------------------------------------------------
                # Date-wise Recovery Analytics
                # --------------------------------------------------

                st.subheader("📅 Date-wise Recovery Analytics")

                date_counts = {}

                for request in recovery_requests:

                    created_at = request.get("created_at")

                    if created_at:
                        request_date = created_at[:10]

                        date_counts[request_date] = (
                            date_counts.get(request_date, 0) + 1
                        )

                if date_counts:

                    sorted_dates = sorted(date_counts.keys())

                    date_chart_data = {
                        "Date": sorted_dates,
                        "Requests": [
                            date_counts[date]
                            for date in sorted_dates
                        ]
                    }

                    st.bar_chart(
                        date_chart_data,
                        x="Date",
                        y="Requests"
                    )

                else:
                    st.info("No date-wise data available.")

                st.divider()

                # --------------------------------------------------
                # Download Recovery Requests as CSV
                # --------------------------------------------------

                st.subheader("📥 Download Recovery Report")

                csv_buffer = io.StringIO()

                csv_writer = csv.writer(csv_buffer)

                csv_writer.writerow([
                    "Request ID",
                    "Item Name",
                    "Item Location",
                    "Requester Name",
                    "Requester Contact",
                    "Message",
                    "Status",
                    "Created At"
                ])

                for request in recovery_requests:

                    item_details = request.get("found_items") or {}

                    csv_writer.writerow([
                        request.get("id"),
                        item_details.get("item_name", "Unknown item"),
                        item_details.get("location", "Not specified"),
                        request.get("requester_name"),
                        request.get("requester_contact"),
                        request.get("message"),
                        request.get("status"),
                        request.get("created_at")
                    ])

                st.download_button(
                    label="📥 Download Recovery Requests CSV",
                    data=csv_buffer.getvalue(),
                    file_name="FindIt_AI_Recovery_Requests.csv",
                    mime="text/csv"
                )

                st.divider()

                # --------------------------------------------------
                # Search and Filter Options
                # --------------------------------------------------

                st.subheader("🔎 Search and Filter Requests")

                search_text = st.text_input(
                    "Search by requester name or item name"
                )

                status_filter = st.selectbox(
                    "Filter by Status",
                    ["All", "Pending", "Contacted", "Returned", "Rejected"]
                )

                filtered_requests = []

                for request in recovery_requests:

                    item_details = request.get("found_items") or {}

                    item_name = item_details.get(
                        "item_name", "Unknown item"
                    )

                    requester_name = request.get(
                        "requester_name", ""
                    )

                    matches_search = (
                        search_text.lower() in requester_name.lower()
                        or search_text.lower() in item_name.lower()
                    )

                    matches_status = (
                        status_filter == "All"
                        or request.get("status") == status_filter
                    )

                    if matches_search and matches_status:
                        filtered_requests.append(request)

                # --------------------------------------------------
                # Display Filtered Requests
                # --------------------------------------------------

                if filtered_requests:

                    st.write(
                        f"Showing {len(filtered_requests)} "
                        f"of {len(recovery_requests)} requests"
                    )

                    for request in filtered_requests:

                        item_details = request.get("found_items") or {}

                        item_name = item_details.get(
                            "item_name", "Unknown item"
                        )

                        item_location = item_details.get(
                            "location", "Not specified"
                        )

                        with st.expander(
                            f"Request #{request['id']} - {item_name}"
                        ):

                            st.write(
                                "**Requester:**",
                                request.get("requester_name", "")
                            )

                            st.write(
                                "**Contact:**",
                                request.get("requester_contact", "")
                            )

                            st.write("**Item:**", item_name)

                            st.write(
                                "**Location:**",
                                item_location
                            )

                            st.write(
                                "**Message:**",
                                request.get("message", "")
                            )

                            st.write(
                                "**Current Status:**",
                                request.get("status", "Pending")
                            )

                            status_options = [
                                "Pending",
                                "Contacted",
                                "Returned",
                                "Rejected"
                            ]

                            current_status = request.get(
                                "status", "Pending"
                            )

                            selected_status = st.selectbox(
                                "Update Request Status",
                                status_options,
                                index=(
                                    status_options.index(current_status)
                                    if current_status in status_options
                                    else 0
                                ),
                                key=f"status_{request['id']}"
                            )

                            if st.button(
                                "Update Status",
                                key=f"update_{request['id']}"
                            ):

                                try:

                                    (
                                        admin_supabase
                                        .table("recovery_requests")
                                        .update({"status": selected_status})
                                        .eq("id", request["id"])
                                        .execute()
                                    )

                                    st.success(
                                        "Status updated successfully!"
                                    )

                                    st.rerun()

                                except Exception as update_error:
                                    st.error("Unable to update status.")
                                    st.code(str(update_error))

                else:
                    st.info(
                        "No recovery requests match your search or filter."
                    )

            except Exception as dashboard_error:
                st.error("Unable to load recovery requests.")
                st.code(str(dashboard_error))

            # --------------------------------------------------
            # Logout
            # --------------------------------------------------

            if st.button("Logout"):

                try:
                    admin_supabase.auth.sign_out()
                except Exception:
                    pass

                st.session_state.admin_logged_in = False
                st.session_state.admin_access_token = None
                st.session_state.admin_refresh_token = None

                st.rerun()


# --------------------------------------------------
# 12. Close Database Connection
# --------------------------------------------------

connection.close()

