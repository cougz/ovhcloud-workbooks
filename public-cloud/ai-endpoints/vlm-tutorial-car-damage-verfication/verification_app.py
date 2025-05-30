import chainlit as cl
# import json # Not directly used in this script but can be kept if your demo uses it
# from datetime import datetime # Not directly used in this script
from verification_demo import CarVerificationDemo # Ensure this file exists and class is defined
import aiofiles
from typing import Dict, List
import re

# Initialize our verification demo
# Ensure CarVerificationDemo is properly defined in verification_demo.py
try:
    demo = CarVerificationDemo()
except NameError:
    # Fallback if CarVerificationDemo is not defined, to prevent immediate crash
    # In a real scenario, ensure verification_demo.py and CarVerificationDemo are correct
    class CarVerificationDemo:
        def verify_vehicle_claims(self, image_data_list, user_claims):
            # Dummy implementation
            print("Warning: CarVerificationDemo not found, using dummy implementation.")
            return {
                "success": True,
                "verification_report": "Dummy Report: VLM did not run.\nClaim: Test\nDoes this match? Yes, it matches dummy claim."
            }
    demo = CarVerificationDemo()


@cl.on_chat_start
async def start_verification():
    """Welcome to the car verification challenge!"""
    cl.user_session.set("user_claims", {})
    cl.user_session.set("current_step", "welcome")
    cl.user_session.set("car_photos", [])
    
    await cl.Message(
        content="""
What **manufacturer/brand** is your car? (BMW, Toyota, Honda, etc.)

*Feel free to tell the truth... or try to trick the AI! 😉*
        """,
        author="Verification AI"
    ).send()

@cl.on_message
async def handle_verification_step(message: cl.Message):
    """Handle each step of the verification process"""
    current_step = cl.user_session.get("current_step", "welcome")
    
    if current_step == "welcome":
        await collect_manufacturer(message.content)
    elif current_step == "model":
        await collect_model(message.content)
    elif current_step == "color":
        await collect_color(message.content)
    elif current_step == "damage":
        await collect_damage(message.content)
    elif current_step == "photos":
        # Check if the message contains actual image files
        image_elements = [
            el for el in message.elements 
            if el.path and ("image" in el.mime if el.mime else el.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
        ]
        if image_elements:
            await collect_photos(message) 
        else:
            # If user sends text instead of photos when photos are expected
            await cl.Message(
                content="📸 Please upload your photos by clicking the paperclip (📎) icon.",
                author="Verification AI"
            ).send()

async def collect_manufacturer(manufacturer: str):
    """Collect car manufacturer claim"""
    manufacturer = manufacturer.strip()
    user_claims = cl.user_session.get("user_claims", {})
    user_claims["manufacturer"] = manufacturer
    cl.user_session.set("user_claims", user_claims)
    cl.user_session.set("current_step", "model")
    
    await cl.Message(
        content=f"""
📝 **Claim #1:** {manufacturer} ✅

Now, what **model** is it? (Camry, Civic, 3 Series, F-150, etc.)
        """,
        author="Verification AI"
    ).send()

async def collect_model(model: str):
    """Collect car model claim"""
    model = model.strip()
    user_claims = cl.user_session.get("user_claims", {})
    user_claims["model"] = model
    cl.user_session.set("user_claims", user_claims)
    cl.user_session.set("current_step", "color")
    
    await cl.Message(
        content=f"""
📝 **Claim #2:** {user_claims.get("manufacturer", "Unknown Make")} {model} ✅

What **color** is your car?
        """,
        author="Verification AI"
    ).send()

async def collect_color(color: str):
    """Collect car color claim"""
    color = color.strip()
    user_claims = cl.user_session.get("user_claims", {})
    user_claims["color"] = color
    cl.user_session.set("user_claims", user_claims)
    cl.user_session.set("current_step", "damage")
    
    await cl.Message(
        content=f"""
📝 **Claim #3:** {color} {user_claims.get("manufacturer", "")} {user_claims.get("model", "")} ✅

Finally, describe any **damage** to your car. If there's no damage, just say "no damage" or "perfect condition".

Examples:
- "dent on front bumper"
- "scratch on driver door"  
- "no damage"
- "minor scratches on rear"
        """,
        author="Verification AI"
    ).send()

async def collect_damage(damage: str):
    """Collect damage claim and move to photo upload"""
    damage = damage.strip()
    user_claims = cl.user_session.get("user_claims", {})
    user_claims["damage"] = damage
    cl.user_session.set("user_claims", user_claims)
    cl.user_session.set("current_step", "photos")
    
    await cl.Message(
        content=f"""
📝 **Claim #4:** {damage} ✅

## 📋 Your Claims Summary:
- **Car:** {user_claims.get("manufacturer", "N/A")} {user_claims.get("model", "N/A")}
- **Color:** {user_claims.get("color", "N/A")}
- **Damage:** {damage}

## 📸 Now the moment of truth!

Upload **at least 3 photos** of your car so the AI can verify if you're telling the truth!

**Upload your photos now!**
        """,
        author="Verification AI"
    ).send()

async def collect_photos(message: cl.Message):
    """Process uploaded photos and start verification"""
    car_photos_session: List[Dict] = cl.user_session.get("car_photos", [])
    new_photos_data: List[Dict] = []
    
    image_elements = [
        el for el in message.elements 
        if el.path and ("image" in el.mime if el.mime else el.name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
    ]

    for element in image_elements:
        try:
            async with aiofiles.open(element.path, 'rb') as f:
                image_byte_data = await f.read()
            new_photos_data.append({"name": element.name, "data": image_byte_data, "path": element.path})
        except Exception as e:
            print(f"Error reading file {element.name}: {e}")
            await cl.Message(content=f"Error processing image {element.name}. Please try again.", author="Verification AI").send()
            # Continue to process other images if any
    
    if not new_photos_data and not image_elements: # If message.elements had non-images, or other error
        if not car_photos_session: # If no photos at all and no new ones successfully read
             await cl.Message(content="No valid image files received. Please upload photos.", author="Verification AI").send()
        return
    
    if new_photos_data:
        car_photos_session.extend(new_photos_data)
        cl.user_session.set("car_photos", car_photos_session)
        
        await cl.Message( 
            content=f"Successfully processed {len(new_photos_data)} new photo(s). Total: {len(car_photos_session)}.",
            author="Verification AI"
        ).send()

    if len(car_photos_session) >= 3:
        await run_verification()
    else:
        await cl.Message(
            content=f"📸 Need {3 - len(car_photos_session)} more photo(s) for verification. Please upload more.",
            author="Verification AI"
        ).send()

async def run_verification():
    """Run the AI verification process"""
    user_claims = cl.user_session.get("user_claims", {})
    car_photos_session = cl.user_session.get("car_photos", [])
    
    processing_msg = cl.Message(content="", author="Verification AI") # Initial empty message
    await processing_msg.send()
    processing_msg.content = "🕵️ **Starting AI Verification...** This may take 30-60 seconds..."
    await processing_msg.update()
    
    try:
        image_data_list = [photo["data"] for photo in car_photos_session[:3]]
        
        async_verify_vehicle_claims = cl.make_async(demo.verify_vehicle_claims)
        verification_result = await async_verify_vehicle_claims(image_data_list, user_claims)
        
        if verification_result and verification_result.get("success"):
            await processing_msg.remove() 
            await show_verification_results(verification_result)
        else:
            error_message = verification_result.get('error', 'Unknown error during verification.')
            await processing_msg.update(content=f"❌ **Verification Failed:** {error_message}")
            cl.user_session.set("current_step", "photos")
            cl.user_session.set("car_photos", []) # Clear photos to allow fresh uploads

    except Exception as e:
        detailed_error_message = f"An unexpected error occurred: {str(e)}"
        print(f"Error during verification: {detailed_error_message}") # Log to console
        await processing_msg.update(content=f"❌ **Error during verification:** {detailed_error_message}")
        cl.user_session.set("current_step", "photos") 
        cl.user_session.set("car_photos", [])

async def show_verification_results(verification_result: Dict):
    """Display the verification results with emojis and improved formatting"""
    user_claims = cl.user_session.get("user_claims", {})
    original_report = verification_result.get("verification_report", "No report generated.")

    # 1. Fix formatting
    # First, normalize very large gaps (3+ newlines to 2)
    formatted_report = re.sub(r'\n{3,}', '\n\n', original_report)
    # After lines ending in '?', ':', or a quote, if followed by \n\n, make it \n
    formatted_report = re.sub(r'([?:"])\n\n', r'\1\n', formatted_report)
    # After lines ending in an alphanumeric character, if followed by \n\n, make it \n
    # This helps compact lists or paragraph breaks where not desired.
    formatted_report = re.sub(r'([a-zA-Z0-9])\n\n', r'\1\n', formatted_report)


    # 2. Add Emojis
    report_lines = formatted_report.split('\n')
    processed_lines = []

    positive_markers = [
        "Yes, it matches their claim",
        "Yes, this matches their claim",
        "This matches their claim accurately",
        "Yes, the photos generally match",
        "The photos provide clear evidence supporting their claims",
        "There are no significant discrepancies",
        "Nothing appears suspicious or inconsistent"
    ]
    negative_markers = [
        "No, it does not match their claim",
        "No, this does not match their claim",
        "This does not match their claim", # Covers "This does not match their claim accurately" too.
        "Discrepancies found:",
        "There are significant discrepancies",
        "The damage is more extensive than just",
        "No, the damage is much more severe than what they claimed",
        "No, the photos do not generally match their claims",
        "Yes, the inconsistencies suggest" # This indicates a problematic finding
    ]

    for line in report_lines:
        stripped_line = line.strip()
        is_positive = any(stripped_line.startswith(marker) for marker in positive_markers)
        is_negative = any(stripped_line.startswith(marker) for marker in negative_markers)

        if is_positive:
            if not stripped_line.startswith("✅"): # Avoid double emojis
                 processed_lines.append(f"✅ {line.lstrip()}")
            else:
                 processed_lines.append(line)
        elif is_negative:
            if not stripped_line.startswith("❌"): # Avoid double emojis
                processed_lines.append(f"❌ {line.lstrip()}")
            else:
                processed_lines.append(line)
        else:
            processed_lines.append(line)

    verification_report_final = "\n".join(processed_lines)
    
    await cl.Message(
        content=f"""# 🎉 Verification Complete!

## 📋 What You Claimed:
- **Manufacturer:** {user_claims.get('manufacturer', 'N/A')}
- **Model:** {user_claims.get('model', 'N/A')}
- **Color:** {user_claims.get('color', 'N/A')}
- **Damage:** {user_claims.get('damage', 'N/A')}

## 🤖 What the AI Actually Saw:""",
        author="Verification AI"
    ).send()
    
    await cl.Message(
        content=verification_report_final,
        author="AI Analysis"
    ).send()

async def ask_for_photos(): # This function is less likely to be called now but kept for completeness
    """Prompt for photo upload"""
    await cl.Message(
        content="📸 **Please upload your car photos!** Click the attachment button (📎) and select at least 3 photos.",
        author="Verification AI"
    ).send()

# To run this application:
# 1. Save this code as a Python file (e.g., `app.py`).
# 2. Make sure you have a `verification_demo.py` file in the same directory
#    with a `CarVerificationDemo` class that has a `verify_vehicle_claims` method.
# 3. Run from your terminal: `chainlit run app.py -w`
