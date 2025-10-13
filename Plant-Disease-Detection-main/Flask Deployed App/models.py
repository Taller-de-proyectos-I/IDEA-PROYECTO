# models.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """Modelo de Usuario con integración para Flask-Login."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    address = db.Column(db.String(255), nullable=True) # Campo para la dirección exacta

    zone_id = db.Column(db.Integer, db.ForeignKey('zone.id'), nullable=False)
    zone = db.relationship('Zone', backref=db.backref('users', lazy=True))

    diagnoses = db.relationship('Diagnosis', backref='user', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Zone(db.Model):
    """Modelo para las zonas geográficas."""
    id = db.Column(db.Integer, primary_key=True)
    district_name = db.Column(db.String(100), nullable=False)
    province_name = db.Column(db.String(100), nullable=False)
    department_name = db.Column(db.String(100), nullable=False, default='Cusco')
    country = db.Column(db.String(100), default='Perú', nullable=False)

    def __repr__(self):
        return f'{self.district_name}, {self.province_name}'

class Diagnosis(db.Model):
    """Modelo para almacenar los diagnósticos."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    image_path = db.Column(db.String(255), nullable=False)
    disease_name = db.Column(db.String(100), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    notes = db.Column(db.Text, nullable=True)
    severity = db.Column(db.String(50), nullable=True)
    image_quality_score = db.Column(db.Float, nullable=True)
    feedback_rating = db.Column(db.Integer, nullable=True)
    feedback_comment = db.Column(db.Text, nullable=True)
