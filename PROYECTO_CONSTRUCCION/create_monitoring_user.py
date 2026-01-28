#!/usr/bin/env python

import os
import sys
import django

# Configurar Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sistema_seguridad.settings')
django.setup()

from django.contrib.auth.models import User

def create_monitoring_user():
    """Crear usuario de monitoreo con permisos de administrador"""
    
    username = 'monitoreo'
    email = 'monitoreo@seguridad.local'
    password = 'Monitoreo2026@Seguro'
    
    # Verificar si el usuario ya existe
    if User.objects.filter(username=username).exists():
        print(f"✓ El usuario '{username}' ya existe")
        user = User.objects.get(username=username)
        # Actualizar contraseña y permisos
        user.set_password(password)
        user.is_staff = True
        user.is_superuser = True
        user.email = email
        user.save()
        print(f"✓ Usuario actualizado con permisos de administrador")
    else:
        # Crear nuevo usuario
        user = User.objects.create_superuser(
            username=username,
            email=email,
            password=password
        )
        print(f"✓ Usuario '{username}' creado exitosamente")
    
    print("\n" + "="*60)
    print("📋 CREDENCIALES DE MONITOREO")
    print("="*60)
    print(f"Usuario:     {username}")
    print(f"Contraseña:  {password}")
    print(f"Email:       {email}")
    print("="*60)
    print("\n✓ Puedes acceder a:")
    print(f"  - Dashboard: http://localhost:8000/dashboard/")
    print(f"  - Admin:     http://localhost:8000/admin/")
    print("\nNOTA: Cambia la contraseña después del primer acceso")
    print("="*60 + "\n")

if __name__ == '__main__':
    create_monitoring_user()
