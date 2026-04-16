import httpx
import os
from django.conf import settings

class SupabaseService:
    @staticmethod
    def get_headers():
        return {
            "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
            "Content-Type": "application/json"
        }

    @staticmethod
    def save_vitals(data):
        # Mapping frontend names to Supabase column names
        supabase_data = {
            "maternal_bp": data.get("maternal_bp"),
            "heart_rate": data.get("maternal_hr"),
            "fetal_hr": data.get("fetal_hr"),
            "spo2": data.get("spo2")
        }
        url = f"{settings.SUPABASE_URL}/rest/v1/patient_cases"
        with httpx.Client() as client:
            response = client.post(url, json=supabase_data, headers=SupabaseService.get_headers())
            if response.status_code < 400:
                try:
                    return response.json()
                except:
                    return {"success": True}
            return {"error": response.text}

    @staticmethod
    def upload_file(file_obj, filename):
        # Supabase Storage Upload
        url = f"{settings.SUPABASE_URL}/storage/v1/object/{settings.SUPABASE_BUCKET}/{filename}"
        headers = {
            "apikey": settings.SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
            "Content-Type": "application/octet-stream"
        }
        with httpx.Client() as client:
            response = client.post(url, content=file_obj.read(), headers=headers)
            if response.status_code < 400:
                return {
                    "success": True,
                    "url": f"{settings.SUPABASE_URL}/storage/v1/object/public/{settings.SUPABASE_BUCKET}/{filename}"
                }
            return {"success": False, "error": response.text}

    @staticmethod
    def save_scan_metadata(metadata):
        url = f"{settings.SUPABASE_URL}/rest/v1/medical_scans"
        with httpx.Client() as client:
            response = client.post(url, json=metadata, headers=SupabaseService.get_headers())
            if response.status_code < 400:
                try:
                    return response.json()
                except:
                    return {"success": True}
            return {"error": response.text}

    @staticmethod
    def get_scan_history():
        url = f"{settings.SUPABASE_URL}/rest/v1/medical_scans?select=*&order=created_at.desc&limit=10"
        with httpx.Client() as client:
            response = client.get(url, headers=SupabaseService.get_headers())
            return response.json() if response.status_code == 200 else []

    @staticmethod
    def verify_login(email, password):
        # Supabase Auth Login
        url = f"{settings.SUPABASE_URL}/auth/v1/token?grant_type=password"
        headers = {
            "apikey": settings.SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
            "Content-Type": "application/json"
        }
        data = {"email": email, "password": password}
        with httpx.Client() as client:
            try:
                response = client.post(url, json=data, headers=headers)
                if response.status_code < 400:
                    return {"success": True, "token": response.json().get("access_token"), "user": response.json().get("user")}
                
                # Detailed error log for the admin
                error_data = response.json()
                msg = error_data.get("error_description", error_data.get("msg", "Login failed"))
                return {"success": False, "error": msg}
            except Exception as e:
                return {"success": False, "error": f"Auth Service unreachable: {str(e)}"}

    @staticmethod
    def signup_user(email, password):
        url = f"{settings.SUPABASE_URL}/auth/v1/signup"
        headers = {
            "apikey": settings.SUPABASE_ANON_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_ANON_KEY}",
            "Content-Type": "application/json"
        }
        data = {"email": email, "password": password}
        with httpx.Client() as client:
            try:
                response = client.post(url, json=data, headers=headers)
                if response.status_code < 400:
                    return {"success": True, "token": response.json().get("access_token"), "user": response.json().get("user")}
                
                error_data = response.json()
                msg = error_data.get("error_description", error_data.get("msg", "Signup failed"))
                return {"success": False, "error": msg}
            except Exception as e:
                return {"success": False, "error": f"Auth Service unreachable: {str(e)}"}

    @staticmethod
    def get_history():
        # Fetch from patient_cases table
        url = f"{settings.SUPABASE_URL}/rest/v1/patient_cases?select=*"
        with httpx.Client() as client:
            response = client.get(url, headers=SupabaseService.get_headers())
            if response.status_code < 400:
                return {"success": True, "records": response.json()}
            return {"success": False, "error": response.text}

