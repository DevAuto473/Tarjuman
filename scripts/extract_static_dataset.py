import cv2
import mediapipe as mp
import os
import csv
import numpy as np

# 1. تهيئة MediaPipe لليدين
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.3  # تم تقليل الحساسية هنا لالتقاط الزوايا الصعبة
)

DATASET_DIR = "dataset"
CSV_FILE = "dataset.csv"

# 2. إنشاء ترويسة ملف CSV
header = ['label']
for i in range(21):
    header.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

print("🚀 جاري بدء معالجة الصور واستخراج الإحداثيات المطبعة...")

with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for letter_folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, letter_folder)
        if not os.path.isdir(folder_path): continue

        success_count = 0
        fail_count = 0

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)
            if image is None: continue

            # ✨ التعديل الجوهري: إزالة الفلاتر وتمرير الصورة الخام كـ RGB مباشرة
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 4. استخراج الإحداثيات
            results = hands.process(rgb_img)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    row_coords = []
                    
                    # نقطة الارتكاز (المعصم)
                    wrist_x = hand_landmarks.landmark[0].x
                    wrist_y = hand_landmarks.landmark[0].y
                    wrist_z = hand_landmarks.landmark[0].z

                    for lm in hand_landmarks.landmark:
                        # طرح إحداثيات المعصم من باقي النقاط
                        nx = lm.x - wrist_x
                        ny = lm.y - wrist_y
                        nz = lm.z - wrist_z
                        row_coords.extend([nx, ny, nz])
                    
                    # ✨ التطبيع (Normalization - Method B) ✨
                    # إيجاد أعلى قيمة مطلقة لتوحيد الحجم (Scaling)
                    max_val = np.max(np.abs(row_coords))
                    if max_val > 0:
                        normalized_coords = (np.array(row_coords) / max_val).tolist()
                    else:
                        normalized_coords = row_coords

                    final_row = [letter_folder] + normalized_coords
                    writer.writerow(final_row)
                    success_count += 1
            else:
                fail_count += 1
                print(f"⚠️ لم يتم اكتشاف يد في: {letter_folder}/{img_name}")

        print(f"📁 حرف [{letter_folder}]: نجاح ({success_count}) | فشل أو تخطي ({fail_count})")

hands.close()
print("✅ تمت المعالجة بنجاح! تم حفظ البيانات في dataset.csv")