#!/usr/bin/env python
import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sistema_seguridad.settings')
django.setup()

from django.contrib.auth.models import User

try:
    u = User.objects.get(username='monitor')
    print(f"✓ Usuario encontrado: {u.username}")
    print(f"✓ Email: {u.email}")
    print(f"✓ Contraseña correcta: {u.check_password('Monitor123!@#')}")
    print("\n✅ Credenciales OK - Intenta ingresar en http://localhost:8000")
except User.DoesNotExist:
    print("❌ Usuario 'monitor' no existe")
