"""
Formularios Django para la aplicación de monitoreo
Validación y procesamiento de datos del usuario
"""

from django import forms
from django.contrib.auth.models import User
from .models import TrainingVideo


class LoginForm(forms.Form):
    """Formulario de autenticación"""
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'Usuario',
            'required': True
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-input',
            'placeholder': 'Contraseña',
            'required': True
        })
    )
    role = forms.ChoiceField(
        choices=[
            ('monitoring', 'Personal de Monitoreo'),
            ('admin', 'Administrador')
        ],
        widget=forms.Select(attrs={
            'class': 'form-input'
        })
    )


class TrainingVideoForm(forms.ModelForm):
    """Formulario para subir videos de entrenamiento"""
    
    class Meta:
        model = TrainingVideo
        fields = ['title', 'behavior_type', 'video', 'description']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-input',
                'placeholder': 'Título del video',
                'required': True
            }),
            'behavior_type': forms.Select(attrs={
                'class': 'form-input',
                'required': True
            }),
            'video': forms.FileInput(attrs={
                'class': 'form-input',
                'accept': 'video/*',
                'required': True
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-textarea',
                'placeholder': 'Descripción del contenido (opcional)',
                'rows': 3
            })
        }
    
    def clean_video(self):
        """Valida el archivo de video"""
        video = self.cleaned_data.get('video')
        
        if video:
            # Validar tamaño (máx 500MB)
            if video.size > 500 * 1024 * 1024:
                raise forms.ValidationError(
                    'El archivo debe ser menor a 500MB'
                )
            
            # Validar formato
            valid_formats = ['mp4', 'avi', 'mov', 'mkv', 'webm']
            file_ext = video.name.split('.')[-1].lower()
            if file_ext not in valid_formats:
                raise forms.ValidationError(
                    f'Formato no soportado. Use: {", ".join(valid_formats)}'
                )
        
        return video
    
    def clean_title(self):
        """Valida el título"""
        title = self.cleaned_data.get('title', '').strip()
        
        if len(title) < 3:
            raise forms.ValidationError(
                'El título debe tener al menos 3 caracteres'
            )
        
        return title


class TrainingBatchForm(forms.Form):
    """Formulario para entrenar el modelo"""
    
    min_samples = forms.IntegerField(
        min_value=2,
        initial=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-input',
            'placeholder': 'Mínimo de muestras',
        })
    )
    test_size = forms.FloatField(
        min_value=0.1,
        max_value=0.5,
        initial=0.2,
        widget=forms.NumberInput(attrs={
            'class': 'form-input',
            'placeholder': 'Proporción de test',
            'step': '0.1'
        })
    )
