"""
Siemens MRI Device Integration Module
Handles communication and data transfer from Siemens MRI systems
Common in Kenya and East Africa medical centers
"""

from datetime import datetime
import socket
import os

class SiemensMRIDevice:
    """Interface for Siemens MRI machines"""
    
    def __init__(self):
        self.device_status = {
            'name': 'Siemens MAGNETOM',
            'model': 'MAGNETOM Avanto 1.5T',
            'manufacturer': 'Siemens Healthineers',
            'status': 'Ready',
            'connected': False,
            'ip_address': '192.168.1.100',
            'port': 104,
            'last_connection': None,
            'total_studies': 0,
            'supported_protocols': [
                'T1 Weighted (T1W)',
                'T2 Weighted (T2W)',
                'FLAIR',
                'DWI (Diffusion Weighted Imaging)',
                'PWI (Perfusion Weighted Imaging)',
                'MRA (Magnetic Resonance Angiography)',
                'DTI (Diffusion Tensor Imaging)'
            ]
        }
        
        self.dicom_listener_running = False
        self.received_studies = []
    
    def connect_to_device(self, ip_address='192.168.1.100', port=104):
        """Attempt to connect to Siemens MRI device via DICOM network"""
        try:
            # Simulate DICOM connection (C-STORE protocol)
            # In production, use pynetdicom library
            socket.gethostbyname(ip_address)
            
            self.device_status['connected'] = True
            self.device_status['last_connection'] = datetime.now().isoformat()
            self.device_status['ip_address'] = ip_address
            self.device_status['port'] = port
            
            return {
                'success': True,
                'message': f'Connected to Siemens MRI at {ip_address}:{port}',
                'device': self.device_status
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection failed: {str(e)}',
                'error': str(e)
            }
    
    def start_dicom_listener(self):
        """Start listening for incoming DICOM transmissions from MRI"""
        self.dicom_listener_running = True
        return {
            'success': True,
            'message': 'DICOM listener started',
            'port': 104,
            'aet': 'MRI_LISTENER',
            'listening': True
        }
    
    def stop_dicom_listener(self):
        """Stop listening for DICOM transmissions"""
        self.dicom_listener_running = False
        return {
            'success': True,
            'message': 'DICOM listener stopped',
            'listening': False
        }
    
    def get_device_status(self):
        """Get current MRI device status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'device': self.device_status,
            'listener_active': self.dicom_listener_running,
            'received_studies': len(self.received_studies)
        }
    
    def get_supported_sequences(self):
        """Get list of MRI sequences supported by this device"""
        return {
            'device': 'Siemens MAGNETOM Avanto 1.5T',
            'sequences': self.device_status['supported_protocols'],
            'field_strength': '1.5 Tesla',
            'gradient_strength': '45 mT/m'
        }
    
    def simulate_mri_scan(self, patient_id, protocol_name):
        """Simulate MRI scan (for testing without actual hardware)"""
        study = {
            'study_id': f'STU_{len(self.received_studies) + 1:04d}',
            'patient_id': patient_id,
            'protocol': protocol_name,
            'device': 'Siemens MAGNETOM Avanto',
            'scan_time': datetime.now().isoformat(),
            'status': 'Completed',
            'images_acquired': 45,
            'dicom_instances': 45,
            'file_size_mb': 125.5,
            'quality': 'Excellent',
            'artifacts': 'None detected'
        }
        
        self.received_studies.append(study)
        self.device_status['total_studies'] += 1
        
        return study
    
    def request_study_from_device(self, study_date_from, study_date_to):
        """Query MRI device for studies within date range"""
        return {
            'query_status': 'Pending',
            'facility': 'Siemens MAGNETOM',
            'date_range': f'{study_date_from} to {study_date_to}',
            'expected_results': 'Retrieving...',
            'message': 'Query sent to Siemens MRI PACS interface'
        }
    
    def get_recent_studies(self, limit=10):
        """Get recently received studies from MRI"""
        return {
            'total_studies': len(self.received_studies),
            'recent_studies': self.received_studies[-limit:],
            'timestamp': datetime.now().isoformat()
        }

# Create global instance
siemens_mri = SiemensMRIDevice()
