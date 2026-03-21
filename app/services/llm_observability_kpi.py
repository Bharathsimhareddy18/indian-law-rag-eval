import os
import logging
from supabase import create_client, Client
from dotenv import load_dotenv
from app.core.config import settings

load_dotenv()
logger = logging.getLogger(__name__)

async def retrive_data_logs(supabase_client: Client):
    
    try:
        response = supabase_client.table("llm_observability") \
                .select("*") \
                .order("created_at", desc=True) \
                .limit(10) \
                .execute()
                
        logger.info("Successfully fetched observability logs from Supabase.")
        return {"message": "Data retrieved successfully", "data": response.data}
        
    except Exception as e:
        logger.error(f"Failed to retrieve observability logs from Supabase: {e}", exc_info=True)
        return {"message": "Failed to retrieve data", "data": [], "error": str(e)}