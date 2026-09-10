"""
Salary Predictor CLI
Usage:
    python predict.py --title "Data Scientist" --exp SE --location US --residence US
"""
import argparse
import os
import joblib
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_artifacts():
    """Load model + encoders + feature engineering artifacts."""
    model          = joblib.load(os.path.join(MODEL_DIR, 'salary_model.pkl'))
    encoders       = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
    title_mean     = joblib.load(os.path.join(MODEL_DIR, 'title_mean.pkl'))
    global_mean    = joblib.load(os.path.join(MODEL_DIR, 'global_mean.pkl'))
    feature_cols   = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
    return model, encoders, title_mean, global_mean, feature_cols


def predict_salary(
    job_title: str,
    experience_level: str,  # EN, MI, SE, EX
    employment_type: str = 'FT',
    employee_residence: str = 'US',
    remote_ratio: int = 0,
    company_location: str = 'US',
    company_size: str = 'M',
    work_year: int = 2025,
):
    model, encoders, title_mean, global_mean, feature_cols = load_artifacts()

    # Compute job_title_avg_salary from training-time lookup
    title_avg = title_mean.get(job_title, global_mean)

    # Build raw row
    row = {
        'work_year': work_year,
        'experience_level': experience_level,
        'employment_type': employment_type,
        'job_title': job_title,
        'employee_residence': employee_residence,
        'remote_ratio': remote_ratio,
        'company_location': company_location,
        'company_size': company_size,
        'job_title_avg_salary': title_avg,
    }

    df = pd.DataFrame([row])

    # Encode categoricals using saved LabelEncoders
    # Handle unseen labels gracefully (fallback to first class)
    for col, le in encoders.items():
        val = str(df.at[0, col])
        if val in le.classes_:
            df[col] = le.transform([val])
        else:
            # Unseen → use closest-matching known label (fallback: first)
            print(f"⚠️  Unknown {col}='{val}', defaulting to '{le.classes_[0]}'")
            df[col] = le.transform([le.classes_[0]])

    # Add interaction features (must match training!)
    df['exp_x_title']  = df['experience_level'] * df['job_title_avg_salary']
    df['year_x_title'] = df['work_year'] * df['job_title_avg_salary']

    # Ensure exact column order
    df = df[feature_cols]

    prediction = model.predict(df)[0]
    return prediction, title_avg


def main():
    parser = argparse.ArgumentParser(description="Predict data science salaries in USD.")
    parser.add_argument('--title',      required=True,  help='Job title (e.g. "Data Scientist")')
    parser.add_argument('--exp',        default='SE',   help='EN/MI/SE/EX (Entry/Mid/Senior/Executive)')
    parser.add_argument('--emp',        default='FT',   help='FT/PT/CT/FL')
    parser.add_argument('--residence',  default='US',   help='Employee residence country code')
    parser.add_argument('--remote',     type=int, default=0, help='0, 50, or 100')
    parser.add_argument('--company_loc',default='US',   help='Company location country code')
    parser.add_argument('--company_size', default='M',  help='S/M/L')
    parser.add_argument('--year',       type=int, default=2025)

    args = parser.parse_args()

    salary, title_avg = predict_salary(
        job_title=args.title,
        experience_level=args.exp,
        employment_type=args.emp,
        employee_residence=args.residence,
        remote_ratio=args.remote,
        company_location=args.company_loc,
        company_size=args.company_size,
        work_year=args.year,
    )

    print("\n" + "=" * 55)
    print(f"  💼 Job Title:      {args.title}")
    print(f"  📊 Experience:     {args.exp}")
    print(f"  📍 Location:       {args.company_loc}")
    print(f"  📈 Title avg:      ${title_avg:,.0f}")
    print(f"  🎯 Predicted:      ${salary:,.0f} USD")
    print("=" * 55 + "\n")


if __name__ == '__main__':
    main()