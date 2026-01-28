"""
Validators - Validaciones reutilizables
"""

from django.core.exceptions import ValidationError


class VideoValidator:
    """Validador para archivos de video"""
    
    ALLOWED_FORMATS = ['mp4', 'avi', 'mov', 'mkv', 'webm']
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    
    @classmethod
    def validate_video_file(cls, file_obj):
        """Valida un archivo de video"""
        errors = []
        
        # Validar tamaño
        if file_obj.size > cls.MAX_FILE_SIZE:
            errors.append(
                f'Archivo muy grande. Máximo: 500MB, Actual: {file_obj.size / (1024*1024):.2f}MB'
            )
        
        # Validar formato
        file_ext = file_obj.name.split('.')[-1].lower()
        if file_ext not in cls.ALLOWED_FORMATS:
            errors.append(
                f'Formato no permitido: {file_ext}. Use: {", ".join(cls.ALLOWED_FORMATS)}'
            )
        
        if errors:
            raise ValidationError(errors)
        
        return True


class TrainingValidator:
    """Validador para datos de entrenamiento"""
    
    MIN_VIDEOS_PER_BEHAVIOR = 5
    MIN_TOTAL_VIDEOS = 20
    
    @classmethod
    def validate_training_dataset(cls, stats):
        """
        Valida que el conjunto de entrenamiento sea adecuado
        
        stats: dict con conteos por comportamiento
        """
        errors = []
        warnings = []
        
        total = sum(stats.values())
        
        # Validaciones críticas
        if total < cls.MIN_TOTAL_VIDEOS:
            errors.append(
                f'Total de videos insuficiente: {total}. Mínimo: {cls.MIN_TOTAL_VIDEOS}'
            )
        
        for behavior, count in stats.items():
            if count < cls.MIN_VIDEOS_PER_BEHAVIOR:
                errors.append(
                    f'{behavior}: {count} videos. Mínimo: {cls.MIN_VIDEOS_PER_BEHAVIOR}'
                )
        
        # Advertencias
        if total < 50:
            warnings.append(
                f'Se recomienda al menos 50 videos para mejor precisión. Actual: {total}'
            )
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


class AuthValidator:
    """Validador para autenticación"""
    
    @staticmethod
    def validate_credentials(username, password):
        """Valida credenciales básicas"""
        errors = []
        
        if not username or not username.strip():
            errors.append('Usuario requerido')
        
        if not password or not password.strip():
            errors.append('Contraseña requerida')
        
        if len(username.strip()) < 3:
            errors.append('Usuario debe tener al menos 3 caracteres')
        
        if len(password.strip()) < 8:
            errors.append('Contraseña debe tener al menos 8 caracteres')
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
