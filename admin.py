from app import app, db, User
from werkzeug.security import generate_password_hash

username = "admin"
password = "admin123"

def create_admin():

    with app.app_context():

        existing = User.query.filter_by(username=username).first()

        if existing:
            print("Admin user already exists")

        else:
            admin = User(
                username=username,
                password=generate_password_hash(password),
                is_admin=True
            )

            db.session.add(admin)
            db.session.commit()

            print("Admin user created successfully!")
            print("Username:", username)
            print("Password:", password)


if __name__ == "__main__":
    create_admin()