import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pandas as pd # Used for DataFrame and CSV export

def _get_message_content(message: Dict[str, Any]) -> str:
    """Extracts human-readable content from a ChatGPT message object."""
    content_obj = message.get('content', {})
    content_type = content_obj.get('content_type')

    if content_type == 'text':
        return " ".join(content_obj.get('parts', [])).strip()
    elif content_type == 'user_editable_context':
        # This often contains system instructions/user profile, might not be direct conversation.
        # You can decide to include it or filter it out. For now, we'll try to extract parts.
        if 'user_profile' in content_obj:
            return f"USER_PROFILE: {content_obj['user_profile'].strip()}"
        if 'user_instructions' in content_obj:
            return f"USER_INSTRUCTIONS: {content_obj['user_instructions'].strip()}"
        if 'parts' in content_obj: # Fallback to parts if they exist
             return " ".join(content_obj.get('parts', [])).strip()
        return "" # Or handle as desired
    elif content_type == 't2uay3k.sj1i4kz': # This often refers to image generation or tool output without explicit text
        # You might want to include a placeholder or parse specific metadata
        return ""
    elif content_type == 'code': # Code interpreter output, usually wrapped in a text 'parts' key
        return " ".join(content_obj.get('parts', [])).strip()
    elif content_type == 'multimodal_text':
        # This is for messages with text and images. We'll only extract text parts.
        text_parts = []
        for part in content_obj.get('parts', []):
            if isinstance(part, dict) and part.get('type') == 'text':
                text_parts.append(part['text'])
            elif isinstance(part, str):
                text_parts.append(part)
        return " ".join(text_parts).strip()
    elif content_type == 'thoughts':
        # This typically contains internal model thoughts, not direct conversation.
        # Can be skipped or captured if desired for analysis of model's thinking process.
        if 'thoughts' in content_obj and isinstance(content_obj['thoughts'], list):
            return "MODEL_THOUGHTS: " + " ".join([t.get('content', '') for t in content_obj['thoughts'] if 'content' in t]).strip()
        if 'summary' in content_obj:
            return "MODEL_THOUGHTS_SUMMARY: " + content_obj['summary'].strip()
        return ""
    elif content_type == 'reasoning_recap':
        return "MODEL_RECAP: " + content_obj.get('content', '').strip()
    elif content_type == 'image_asset_pointer':
        # This indicates an image, no text content directly.
        return ""
    else:
        # Catch-all for other content types not explicitly handled
        return str(content_obj)


def normalize_chatgpt_json(file_path: str) -> List[Dict[str, Any]]:
    """Parses ChatGPT JSON and flattens it into a list of normalized message dicts."""
    normalized_messages = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: ChatGPT file not found at {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding ChatGPT JSON from {file_path}: {e}")
        return []

    for convo in data:
        convo_id = convo.get('id')
        convo_title = convo.get('title', 'Untitled Chat')
        
        # Collect all message nodes from the 'mapping' object
        # The 'mapping' keys are message_ids, and values are message objects or metadata.
        messages_in_convo = []
        for node_id, node_data in convo.get('mapping', {}).items():
            if 'message' in node_data and node_data['message']:
                msg = node_data['message']
                # Only include messages that have a discernible role and content
                if msg.get('author') and msg.get('author').get('role'):
                    messages_in_convo.append(msg)

        # Sort messages by creation time to maintain conversation flow
        # Filter out messages with no create_time if they cause issues, or assign convo_time
        messages_in_convo_sorted = sorted(
            [msg for msg in messages_in_convo if msg.get('create_time')],
            key=lambda x: x['create_time']
        )
        
        # If no messages have create_time but there are messages,
        # we might use the conversation's create_time or just process in mapping order.
        if not messages_in_convo_sorted and messages_in_convo:
            messages_in_convo_sorted = messages_in_convo # Fallback to original order


        for msg in messages_in_convo_sorted:
            role = msg['author']['role']
            role_label = 'User' if role == 'user' else ('Assistant' if role == 'assistant' else role.capitalize())

            content = _get_message_content(msg)
            
            timestamp_unix = msg.get('create_time')
            if timestamp_unix is not None:
                try:
                    dt = datetime.fromtimestamp(timestamp_unix, tz=timezone.utc)
                    timestamp_iso = dt.isoformat().replace('+00:00', 'Z')
                except Exception:
                    timestamp_iso = None
            else:
                timestamp_iso = None

            # Filter out empty content or non-conversational messages unless explicitly desired
            if content.strip(): # Only include messages that actually have text content
                normalized_messages.append({
                    "Source AI": "ChatGPT",
                    "Conversation ID": convo_id,
                    "Conversation Title": convo_title,
                    "Message ID": msg.get('id'),
                    "Timestamp": timestamp_iso,
                    "Role": role_label,
                    "Content": content,
                    "Word Count": len(content.split())
                })
    return normalized_messages

def normalize_claude_json(file_path: str) -> List[Dict[str, Any]]:
    """Parses Claude JSON export and flattens it into a list of normalized message dicts."""
    normalized_messages = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Claude file not found at {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding Claude JSON from {file_path}: {e}")
        return []

    for conversation in data:
        convo_uuid = conversation.get('uuid')
        convo_name = conversation.get('name', 'Untitled Chat')

        chat_messages = conversation.get('chat_messages', [])
        if not chat_messages:
            continue # Skip empty conversations

        for msg in chat_messages:
            message_uuid = msg.get('uuid')
            sender = msg.get('sender')
            
            # Map Claude's sender to a standardized Role
            role_label = 'User' if sender == 'human' else ('Assistant' if sender == 'assistant' else sender.capitalize())

            # Extract content. The 'content' field is an array of objects.
            message_content_parts = []
            for content_block in msg.get('content', []):
                if content_block.get('type') == 'text' and 'text' in content_block:
                    message_content_parts.append(content_block['text'])
                # If you want to capture tool_use, images, etc., add more logic here.
                # For a general text-based knowledge base, we focus on 'text' type.
            
            full_message_content = " ".join(message_content_parts).strip()
            
            # Use the 'text' field as a fallback or primary content if it exists and content parts are empty
            if not full_message_content and msg.get('text'):
                full_message_content = msg['text'].strip()

            created_at_raw = msg.get('created_at')
            if created_at_raw:
                try:
                    # Normalize to timezone-aware UTC and render as Z
                    dt = datetime.fromisoformat(str(created_at_raw).replace('Z', '+00:00'))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    created_at_iso = dt.isoformat().replace('+00:00', 'Z')
                except Exception:
                    created_at_iso = str(created_at_raw)
            else:
                created_at_iso = None

            if full_message_content.strip(): # Only add messages with actual content
                normalized_messages.append({
                    "Source AI": "Claude",
                    "Conversation ID": convo_uuid,
                    "Conversation Title": convo_name,
                    "Message ID": message_uuid,
                    "Timestamp": created_at_iso,
                    "Role": role_label,
                    "Content": full_message_content,
                    "Word Count": len(full_message_content.split())
                })
    return normalized_messages

def merge_and_save_normalized_data(chatgpt_file: str, claude_file: str, output_csv_path: str):
    """
    Normalizes data from both sources, merges it, and saves to a CSV.
    This CSV can then be imported directly into Notion.
    """
    chatgpt_data = normalize_chatgpt_json(chatgpt_file)
    claude_data = normalize_claude_json(claude_file)

    all_normalized_data = chatgpt_data + claude_data

    # Sort all messages chronologically
    # Convert 'Timestamp' to datetime objects for proper sorting, handle potential None
    from datetime import timezone
    for item in all_normalized_data:
        if item.get('Timestamp'):
            try:
                # Parse timestamp and ensure it's timezone-aware
                dt = datetime.fromisoformat(item['Timestamp'].replace('Z', '+00:00'))
                # If timezone-naive, assume UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                item['Sort_Key'] = dt
            except ValueError:
                item['Sort_Key'] = datetime.min.replace(tzinfo=timezone.utc) # Assign a minimum date if parsing fails
        else:
            item['Sort_Key'] = datetime.min.replace(tzinfo=timezone.utc) # Messages with no timestamp go to the start

    all_normalized_data.sort(key=lambda x: x['Sort_Key'])

    # Remove the temporary sort key before creating DataFrame
    for item in all_normalized_data:
        item.pop('Sort_Key', None)

    df = pd.DataFrame(all_normalized_data)
    
    # Ensure all desired columns are present, even if some rows don't have them
    # This prevents issues with Notion import if columns are missing for some entries
    required_columns = ["Source AI", "Conversation ID", "Conversation Title", "Message ID", "Timestamp", "Role", "Content", "Word Count"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = None # Add missing column with None values

    # Reorder columns for consistency
    df = df[required_columns]

    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig') # utf-8-sig for Excel/Notion compatibility
    print(f"Normalized and merged data saved to {output_csv_path}")

# --- Main execution block ---
if __name__ == "__main__":
    # Ensure you replace these with the actual paths to your exported JSON files
    # For user-provided files, they are usually in the sandbox at /mnt/data/
    chatgpt_export_file = './chatgpt_conversations.json' # Assuming user named it this or similar
    claude_export_file = './claude_conversations.json'   # Assuming user named it this or similar
    output_csv = 'combined_ai_chat_history.csv'

    # Important: You might need to manually rename the uploaded files in the /mnt/data/ directory
    # if their names are not 'chatgpt_conversations.json' and 'claude_conversations.json'
    # The system provides file names 'conversations.json' for both, so they will need renaming
    # or you need to specify the correct uploaded file name like 'uploaded_chatgpt.json' etc.

    # Example: if you uploaded both as 'conversations.json', you'd need to manually
    # distinguish them by their content or rename them before running this script fully.
    # For the purpose of this response, I'm assuming they are distinctly named.

    merge_and_save_normalized_data(chatgpt_export_file, claude_export_file, output_csv)

    print(f"\nYour combined chat history is ready for import into Notion: {output_csv}")
    print("Ensure Notion database column names exactly match these (case-sensitive):")
    print("Source AI, Conversation ID, Conversation Title, Message ID, Timestamp, Role, Content, Word Count")
    print("\nNext, you can manually import this CSV file into a Notion database.")
    print("For best results, create a new database in Notion and import the CSV into it.")
    print("Then you can use Notion AI for cross-chat analysis or expose data via Notion's API.")