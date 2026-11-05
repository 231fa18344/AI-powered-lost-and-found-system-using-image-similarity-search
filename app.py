
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

# LOGIN STATE

if "student_logged_in" not in st.session_state:
    st.session_state.student_logged_in = False

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if "selected_page" not in st.session_state:
    st.session_state.selected_page = None


# --------------------------------------------------
# PREMIUM MOBILE-FRIENDLY DESIGN
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #f5f8fc;
}

div.stButton > button {
    min-height: 58px;
    border-radius: 14px;
    font-size: 16px;
    font-weight: 700;
    border: 1px solid #cbd5e1;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    transform: translateY(-3px);
    border-color: #247b70;
    color: #247b70;
}

div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    background: linear-gradient(145deg, #101f36, #142b48);
    padding: 15px;
    border-radius: 18px;
    border: 1px solid #344966;
    width: 100%;
    box-sizing: border-box;
}

div[data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}

div[data-testid="stRadio"] label > div:first-child {
    display: none !important;
}

div[data-testid="stRadio"] label {
    display: flex !important;
    justify-content: center;
    align-items: center;
    background: #1b2d49;
    border: 1px solid #344966;
    border-radius: 13px;
    padding: 12px 7px;
    min-height: 50px;
    width: 100%;
    box-sizing: border-box;
    margin: 0 !important;
    cursor: pointer;
}

div[data-testid="stRadio"] label p {
    color: #f1f5f9 !important;
    font-size: 13px;
    font-weight: 600;
    text-align: center;
    margin: 0;
}

div[data-testid="stRadio"] label:hover {
    background: #293e5e;
    border-color: #5b789f;
}

div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, #397d72, #28695f);
    border: 1px solid #69c5ad;
}

div[data-testid="stRadio"] label:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 700;
}

@media screen and (max-width: 768px) {

    div[data-testid="stRadio"] > div[role="radiogroup"] {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
        padding: 10px;
    }

    div[data-testid="stRadio"] label {
        min-height: 46px;
        padding: 9px 4px;
    }

    div[data-testid="stRadio"] label p {
        font-size: 11px;
    }

    section.main > div {
        padding-left: 12px;
        padding-right: 12px;
    }

    div[data-testid="stImage"] img {
        max-width: 100%;
        height: auto;
        object-fit: contain;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        width: 100%;
        box-sizing: border-box;
    }
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# FIRST SCREEN - STUDENT / ADMIN LOGIN
# --------------------------------------------------

if not (
    st.session_state.student_logged_in
    or st.session_state.admin_logged_in
):

    st.markdown(
        "<div style='height:70px'></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h1 style='text-align:center;color:#247b70;'>"
        "🔎 FindIt AI</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='text-align:center;'>"
        "Welcome to Smart Campus Lost &amp; Found"
        "</h3>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center;color:#64748b;'>"
        "Lost something on campus?<br>"
        "Find your belongings using Artificial Intelligence.<br>"
        "Report found items and reconnect them with their owners."
        "</p>",
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    empty_left, center_col, empty_right = st.columns([1, 2, 1])

    with center_col:

        student_col, admin_col = st.columns(2)

        with student_col:

            if st.button(
                "👤 Student Login",
                use_container_width=True,
                key="welcome_student_button"
            ):

                st.session_state.selected_page = "👤 Student Login"
                st.rerun()

        with admin_col:

            if st.button(
                "🔐 Admin Login",
                use_container_width=True,
                key="welcome_admin_button"
            ):

                st.session_state.selected_page = "🔐 Admin"
                st.rerun()

    st.write("")

    st.markdown(
        "<p style='text-align:center;color:#64748b;"
        "font-size:13px;'>"
        "AI-powered campus lost-and-found platform"
        "</p>",
        unsafe_allow_html=True
    )

    selected_page = st.session_state.selected_page

    if selected_page is None:
        st.stop()


# --------------------------------------------------
# AFTER LOGIN - MAIN NAVIGATION
# --------------------------------------------------

else:

    st.title("🔎 FindIt AI")

    # ----------------------------------------------
    # STUDENT NAVIGATION
    # ----------------------------------------------

    if st.session_state.student_logged_in:

        st.caption("👤 Student Account")

        navigation_options = [
            "🏠 Home",
            "📦 Report Item",
            "🔍 Search Items",
            "🤖 AI Image Search",
            "📋 All Found Items",
            "📩 Recovery Requests",
            "📤 Share",
            "👤 Student Login"
        ]

    # ----------------------------------------------
    # ADMIN NAVIGATION
    # ----------------------------------------------

    elif st.session_state.admin_logged_in:

        st.caption("🔐 Administrator Account")

        navigation_options = [
            "🏠 Home",
            "📦 Report Item",
            "🔍 Search Items",
            "🤖 AI Image Search",
            "📋 All Found Items",
            "📩 Recovery Requests",
            "📤 Share",
            "👤 Student Login",
            "🔐 Admin"
        ]

    # ----------------------------------------------
    # RESET NAVIGATION WHEN ROLE CHANGES
    # ----------------------------------------------

    if (
        "previous_navigation_options" not in st.session_state
        or st.session_state.previous_navigation_options
        != navigation_options
    ):

        st.session_state.previous_navigation_options = navigation_options

        if "main_navigation" in st.session_state:
            del st.session_state["main_navigation"]

    # ----------------------------------------------
    # DISPLAY NAVIGATION
    # ----------------------------------------------

    selected_page = st.radio(
        "Navigation",
        navigation_options,
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation"
    )

    st.divider()
# --------------------------------------------------
# 6. REPORT A FOUND ITEM
# --------------------------------------------------

if selected_page == "📦 Report Item":

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

            st.subheader("Student Login")

            login_email = st.text_input(
                "Email",
                key="student_login_email"
            )

            login_password = st.text_input(
                "Password",
                type="password",
                key="student_login_password"
            )

            if st.button("Login", key="student_login_button"):

                if not login_email or not login_password:
                    st.warning("Please enter email and password.")

                else:

                    try:

                        login_response = supabase.auth.sign_in_with_password({
                            "email": login_email,
                            "password": login_password
                        })

                        if login_response.user:

                            st.session_state.student_logged_in = True
                            st.session_state.student_user_id = login_response.user.id
                            st.session_state.student_access_token = login_response.session.access_token
                            st.session_state.student_refresh_token = login_response.session.refresh_token

                            st.success("Login successful!")
                            st.rerun()

                    except Exception as login_error:

                        st.error("Invalid email or password.")
                        st.code(str(login_error))

        # ------------------------------------------
        # STUDENT SIGN UP
        # ------------------------------------------

        with signup_tab:

            st.subheader("Create Student Account")

            signup_name = st.text_input(
                "Full Name",
                key="student_signup_name"
            )

            signup_email = st.text_input(
                "Email",
                key="student_signup_email"
            )

            signup_phone = st.text_input(
                "Phone Number",
                key="student_signup_phone"
            )

            signup_password = st.text_input(
                "Password",
                type="password",
                key="student_signup_password"
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                key="student_confirm_password"
            )

            if st.button("Create Account", key="student_signup_button"):

                if not signup_name or not signup_email or not signup_phone or not signup_password:
                    st.warning("Please fill all fields.")

                elif len(signup_phone) != 10 or not signup_phone.isdigit():
                    st.warning("Enter a valid 10-digit phone number.")

                elif len(signup_password) < 6:
                    st.warning("Password must contain at least 6 characters.")

                elif signup_password != confirm_password:
                    st.warning("Passwords do not match.")

                else:

                    try:

                        signup_response = supabase.auth.sign_up({
                            "email": signup_email,
                            "password": signup_password
                        })

                        if signup_response.user:

                            student_uid = signup_response.user.id

                            supabase.table("student_profiles").upsert({
                                "id": student_uid,
                                "full_name": signup_name,
                                "email": signup_email,
                                "phone": signup_phone
                            }).execute()

                            st.success(
                                "Account created successfully! Please login."
                            )

                    except Exception as signup_error:

                        st.error("Unable to create account.")
                        st.code(str(signup_error))

    else:

        # ------------------------------------------
        # STUDENT DASHBOARD
        # ------------------------------------------

        st.success("You are logged in!")

        student_uid = st.session_state.student_user_id

        try:

            student_supabase = create_client(
                SUPABASE_URL,
                SUPABASE_KEY
            )

            student_supabase.auth.set_session(
                st.session_state.student_access_token,
                st.session_state.student_refresh_token
            )

            profile_response = (
                student_supabase.table("student_profiles")
                .select("*")
                .eq("id", student_uid)
                .execute()
            )

            student_profile = (
                profile_response.data[0]
                if profile_response.data
                else {}
            )

            st.subheader("👤 My Profile")

            profile_name = st.text_input(
                "Full Name",
                value=student_profile.get("full_name", "")
            )

            profile_email = st.text_input(
                "Email",
                value=student_profile.get("email", ""),
                disabled=True
            )

            profile_phone = st.text_input(
                "Phone Number",
                value=student_profile.get("phone", "")
            )

            if st.button("Update Profile"):

                student_supabase.table("student_profiles").upsert({
                    "id": student_uid,
                    "full_name": profile_name,
                    "email": profile_email,
                    "phone": profile_phone
                }).execute()

                st.success("Profile updated successfully!")
                st.rerun()

            st.divider()

            # ------------------------------------------
            # MY NOTIFICATIONS
            # ------------------------------------------

            st.subheader("🔔 My Notifications")

            try:

                notifications_response = (
                    student_supabase.table("notifications")
                    .select("id, title, message, is_read, created_at")
                    .eq("user_id", student_uid)
                    .order("created_at", desc=True)
                    .execute()
                )

                my_notifications = notifications_response.data or []

                unread_count = sum(
                    1 for notification in my_notifications
                    if not notification.get("is_read", False)
                )

                st.info(
                    f"🔔 You have {unread_count} unread notification(s)."
                )

                if my_notifications:

                    for notification in my_notifications:

                        with st.container(border=True):

                            if notification.get("is_read", False):
                                st.write(
                                    f"📩 **{notification.get('title', 'Notification')}**"
                                )
                            else:
                                st.write(
                                    f"🔵 **{notification.get('title', 'Notification')}**"
                                )

                            st.write(notification.get("message", ""))

                            st.caption(
                                "Received: "
                                + str(notification.get("created_at", ""))[:19]
                            )

                            if not notification.get("is_read", False):

                                if st.button(
                                    "Mark as Read",
                                    key=f"read_notification_{notification['id']}"
                                ):

                                    student_supabase.table("notifications").update({
                                        "is_read": True
                                    }).eq(
                                        "id", notification["id"]
                                    ).eq(
                                        "user_id", student_uid
                                    ).execute()

                                    st.rerun()

                else:

                    st.info("You don't have any notifications yet.")

            except Exception as notification_error:

                st.error("Unable to load notifications.")
                st.code(str(notification_error))

            st.divider()

            # ------------------------------------------
            # MY RECOVERY REQUESTS
            # ------------------------------------------

            st.subheader("📦 My Recovery Requests")

            requests_response = (
                student_supabase.table("recovery_requests")
                .select("*")
                .eq("student_id", student_uid)
                .order("created_at", desc=True)
                .execute()
            )

            my_requests = requests_response.data or []

            if my_requests:

                for request in my_requests:

                    with st.container(border=True):

                        st.write(
                            f"**Request ID:** {request.get('id', '')}"
                        )

                        st.write(
                            f"**Status:** {request.get('status', 'Pending')}"
                        )

                        item_details = {}

                        try:

                            item_response = (
                                student_supabase.table("found_items")
                                .select("*")
                                .eq("id", request.get("found_item_id"))
                                .execute()
                            )

                            if item_response.data:
                                item_details = item_response.data[0]

                        except Exception:
                            pass

                        st.write(
                            f"**Item:** {item_details.get('item_name', 'Item details unavailable')}"
                        )

                        st.write(
                            f"**Location:** {item_details.get('location', 'Not specified')}"
                        )

                        # SHOW FOUND ITEM PHOTO

                        item_image = item_details.get("image_path")

                        if item_image:

                            try:
                                st.image(item_image, width=200)
                            except Exception:
                                pass

                        st.write(
                            f"**Submitted:** {str(request.get('created_at', ''))[:19]}"
                        )

            else:

                st.info("You have not submitted any recovery requests yet.")

            st.divider()

            # ------------------------------------------
            # STUDENT LOGOUT
            # ------------------------------------------

            if st.button("🚪 Logout", key="student_logout_button"):

                try:
                    student_supabase.auth.sign_out()
                except Exception:
                    pass

                st.session_state.student_logged_in = False
                st.session_state.student_access_token = None
                st.session_state.student_refresh_token = None
                st.session_state.student_user_id = None

                st.success("Logged out successfully!")
                st.rerun()

        except Exception as student_error:

            st.error("Unable to load student dashboard.")
            st.code(str(student_error))


# ==================================================
# ADMIN LOGIN AND DASHBOARD
# ==================================================

elif selected_page == "🔐 Admin":

    ADMIN_UID = "1fa8098e-27be-46c1-9916-1e20506f9fa5"

    st.title("🔎 FindIt AI")

    st.header("🔐 Admin Login")

    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False

    if "admin_access_token" not in st.session_state:
        st.session_state.admin_access_token = None

    if "admin_refresh_token" not in st.session_state:
        st.session_state.admin_refresh_token = None

    if not st.session_state.admin_logged_in:

        admin_email = st.text_input(
            "Admin Email",
            key="admin_email"
        )

        admin_password = st.text_input(
            "Admin Password",
            type="password",
            key="admin_password"
        )

        if st.button("Admin Login", key="admin_login_button"):

            try:

                admin_response = supabase.auth.sign_in_with_password({
                    "email": admin_email,
                    "password": admin_password
                })

                if admin_response.user:

                    if admin_response.user.id == ADMIN_UID:

                        st.session_state.admin_logged_in = True
                        st.session_state.admin_access_token = admin_response.session.access_token
                        st.session_state.admin_refresh_token = admin_response.session.refresh_token

                        st.success("Admin login successful!")
                        st.rerun()

                    else:

                        supabase.auth.sign_out()
                        st.error("You are not authorized as Admin.")

            except Exception as admin_error:

                st.error("Invalid admin credentials.")
                st.code(str(admin_error))

    else:

        st.success("Welcome Admin!")

        admin_supabase = create_client(
            SUPABASE_URL,
            SUPABASE_KEY
        )

        admin_supabase.auth.set_session(
            st.session_state.admin_access_token,
            st.session_state.admin_refresh_token
        )

        # ------------------------------------------
        # LOAD DASHBOARD DATA
        # ------------------------------------------

        recovery_response = (
            admin_supabase.table("recovery_requests")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )

        found_response = (
            admin_supabase.table("found_items")
            .select("*")
            .execute()
        )

        students_response = (
            admin_supabase.table("student_profiles")
            .select("*")
            .execute()
        )

        all_requests = recovery_response.data or []
        all_found_items = found_response.data or []
        all_students = students_response.data or []

        found_items_map = {
            item.get("id"): item
            for item in all_found_items
        }

        students_map = {
            student.get("id"): student
            for student in all_students
        }

        # ------------------------------------------
        # DASHBOARD METRICS
        # ------------------------------------------

        st.header("📊 Admin Dashboard")

        total_requests = len(all_requests)
        total_items = len(all_found_items)
        total_students = len(all_students)

        pending_requests = sum(
            1 for request in all_requests
            if request.get("status") == "Pending"
        )

        contacted_requests = sum(
            1 for request in all_requests
            if request.get("status") == "Contacted"
        )

        returned_requests = sum(
            1 for request in all_requests
            if request.get("status") == "Returned"
        )

        rejected_requests = sum(
            1 for request in all_requests
            if request.get("status") == "Rejected"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Requests", total_requests)
        col2.metric("Found Items", total_items)
        col3.metric("Students", total_students)
        col4.metric("Pending Requests", pending_requests)

        st.divider()

        # ------------------------------------------
        # REQUEST STATUS SUMMARY
        # ------------------------------------------

        st.subheader("📌 Request Status Summary")

        status_col1, status_col2, status_col3, status_col4 = st.columns(4)

        status_col1.metric("Pending", pending_requests)
        status_col2.metric("Contacted", contacted_requests)
        status_col3.metric("Returned", returned_requests)
        status_col4.metric("Rejected", rejected_requests)

        st.divider()

        # ------------------------------------------
        # REQUEST DETAILS
        # ------------------------------------------

        st.subheader("📋 All Recovery Requests")

        if all_requests:

            for request in all_requests:

                request_id = request.get("id")
                student_id = request.get("student_id")
                found_item_id = request.get("found_item_id")

                student_details = students_map.get(student_id, {})
                item_details = found_items_map.get(found_item_id, {})

                with st.container(border=True):

                    st.write(f"**Request ID:** {request_id}")

                    st.write(
                        f"**Student:** {student_details.get('full_name', 'Unknown')}"
                    )

                    st.write(
                        f"**Email:** {student_details.get('email', 'Not available')}"
                    )

                    st.write(
                        f"**Phone:** {student_details.get('phone', 'Not available')}"
                    )

                    st.write(
                        f"**Item:** {item_details.get('item_name', 'Unknown')}"
                    )

                    st.write(
                        f"**Location:** {item_details.get('location', 'Not specified')}"
                    )

                    st.write(
                        f"**Current Status:** {request.get('status', 'Pending')}"
                    )

                    st.write(
                        f"**Submitted:** {str(request.get('created_at', ''))[:19]}"
                    )

                    # ----------------------------------
                    # UPDATE REQUEST STATUS
                    # ----------------------------------

                    current_status = request.get("status", "Pending")

                    status_options = [
                        "Pending",
                        "Contacted",
                        "Returned",
                        "Rejected"
                    ]

                    selected_status = st.selectbox(
                        "Update Status",
                        status_options,
                        index=(
                            status_options.index(current_status)
                            if current_status in status_options
                            else 0
                        ),
                        key=f"status_{request_id}"
                    )

                    if st.button(
                        "Update Status",
                        key=f"update_status_{request_id}"
                    ):

                        admin_supabase.table("recovery_requests").update({
                            "status": selected_status
                        }).eq(
                            "id", request_id
                        ).execute()

                        st.success("Request status updated successfully!")
                        st.rerun()

        else:

            st.info("No recovery requests available.")

        st.divider()

        # ------------------------------------------
        # CSV DOWNLOAD
        # ------------------------------------------

        st.subheader("📥 Download Recovery Requests")

        if all_requests:

            import pandas as pd

            export_rows = []

            for request in all_requests:

                student_details = students_map.get(
                    request.get("student_id"), {}
                )

                item_details = found_items_map.get(
                    request.get("found_item_id"), {}
                )

                export_rows.append({
                    "Request ID": request.get("id"),
                    "Student Name": student_details.get("full_name", ""),
                    "Email": student_details.get("email", ""),
                    "Phone": student_details.get("phone", ""),
                    "Item Name": item_details.get("item_name", ""),
                    "Location": item_details.get("location", ""),
                    "Status": request.get("status", ""),
                    "Created At": request.get("created_at", "")
                })

            export_df = pd.DataFrame(export_rows)

            csv_data = export_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name="recovery_requests.csv",
                mime="text/csv"
            )

        else:

            st.info("No data available to download.")

        st.divider()

        # ------------------------------------------
        # ADMIN LOGOUT
        # ------------------------------------------

        if st.button("🚪 Admin Logout", key="admin_logout_button"):

            try:
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
