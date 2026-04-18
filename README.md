# HeartAttackPredictor

It is a heart Attack prediction web app which tells us about the Heart Diseases
We provide it details related heart problems then it tells us Attack risk
The name is HeartGuard

## Setup Instructions

### Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and configure your settings
4. Run the application:
   ```bash
   python app.py
   ```

### Database Setup

#### Local Development (SQLite)
The app defaults to SQLite for local development. No additional setup required.

#### Production (Render PostgreSQL)

1. **Create a PostgreSQL database on Render:**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "PostgreSQL"
   - Choose a name and region
   - Click "Create Database"

2. **Get your database URL:**
   - In your Render database dashboard, go to "Connections"
   - Copy the "External Database URL" (starts with `postgresql://`)

3. **Configure environment variables:**
   - In your Render web service settings, add the following environment variables:
     - `DATABASE_URL`: Your PostgreSQL connection string (External Database URL)
     - `SECRET_KEY`: A random secret key for Flask sessions (generate a secure one)
     - `MODEL_PATH`: `heart_attack_model.pkl`
     - `FLASK_DEBUG`: `False` (for production)

4. **Deploy to Render:**
   - Push your code to GitHub
   - Create a new Render Web Service
   - Connect your GitHub repository
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `gunicorn app:app`
   - Add your environment variables in the "Environment" section
   - Deploy!

**Note:** Database connections from your local machine will fail (connection timeout) because Render databases are only accessible from within Render's network. This is normal - your app will connect successfully when deployed on Render.

### Environment Variables

- `SECRET_KEY`: Flask secret key for sessions
- `DATABASE_URL`: Database connection string
  - Local: `sqlite:///heart_data.db`
  - Render: `postgresql://username:password@host:5432/database`
- `MODEL_PATH`: Path to the ML model file
- `FLASK_DEBUG`: Debug mode (True/False)

### Switching Between Local and Production

**For Local Development:**
```bash
# Use SQLite (default in .env.example)
DATABASE_URL=sqlite:///heart_data.db
FLASK_DEBUG=True
```

**For Production (Render):**
```bash
# Use PostgreSQL URL from Render dashboard
DATABASE_URL=postgresql://your-render-database-url
FLASK_DEBUG=False
SECRET_KEY=your-secure-random-key
```

### Testing Database Connection

Run the test script to verify your database setup:
```bash
python test_db.py
```

### File Structure

```
HeartAttackPredictor/
├── app.py                 # Main Flask application
├── model.py              # ML model training script
├── heart.csv            # Dataset
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables (not in git)
├── .env.example         # Environment template
├── .gitignore           # Git ignore rules
├── static/              # Static files (CSS, JS, images)
├── templates/           # HTML templates
│   ├── index.html
│   ├── login.html
│   └── profile.html
└── README.md
```
