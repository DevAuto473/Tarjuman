import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# =====================================================================
MODEL_OUTPUT = 'sign_model.pkl'
N_TREES      = 100
MAX_DEPTH    = 15      # تحديد عمق الشجرة لمنع الحفظ العشوائي
MIN_SAMPLES  = 5       # إجبار النموذج على تجاهل الحركات الشاذة الفردية

print("🚀 جاري قراءة البيانات...")
df = pd.read_csv('dataset.csv')

X = df.drop('label', axis=1)
y = df['label']

print(f"📊 عدد العينات الكلي: {len(df):,} | عدد التصنيفات: {y.nunique()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nجاري تدريب النموذج...")
print(f"   ├─ الأشجار    : {N_TREES}")
print(f"   ├─ أقصى عمق   : {MAX_DEPTH}")
print(f"   └─ البيانات   : {len(X_train):,} عينة تدريب\n")

model = RandomForestClassifier(
    n_estimators=N_TREES,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f" دقة النموذج: {accuracy * 100:.2f}%")
print("\n تقرير الدقة لكل تصنيف:")
print(classification_report(y_test, y_pred))

with open(MODEL_OUTPUT, 'wb') as f:
    pickle.dump(model, f)

size_mb = os.path.getsize(MODEL_OUTPUT) / (1024 * 1024)

print(f"\n تم حفظ النموذج: {MODEL_OUTPUT}")
print(f"   └─ الحجم: {size_mb:.1f} MB")