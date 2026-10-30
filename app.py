
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
# 10. Safe Item Recovery Request
# --------------------------------------------------

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

ADMIN_UID = "1fa8098e-27be-46c1-9916-1e20506f9fa5"

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
                    or request["status"] == status_filter
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
                            request["requester_name"]
                        )

                        st.write(
                            "**Contact:**",
                            request["requester_contact"]
                        )

                        st.write("**Item:**", item_name)

                        st.write(
                            "**Location:**",
                            item_location
                        )

                        st.write(
                            "**Message:**",
                            request["message"]
                        )

                        st.write(
                            "**Current Status:**",
                            request["status"]
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
