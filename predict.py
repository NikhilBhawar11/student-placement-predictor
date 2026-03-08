import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

print("Enter Student Details")

cgpa = float(input("CGPA: "))
internships = int(input("Internships: "))
projects = int(input("Projects: "))
workshops = int(input("Workshops/Certifications: "))
aptitude = int(input("Aptitude Test Score: "))
softskills = int(input("Soft Skills Rating (1-10): "))
extra = int(input("Extracurricular Activities (1=Yes,0=No): "))
training = int(input("Placement Training (1=Yes,0=No): "))
ssc = int(input("SSC Marks: "))
hsc = int(input("HSC Marks: "))

data = np.array([[cgpa, internships, projects, workshops,
                  aptitude, softskills, extra, training,
                  ssc, hsc]])

prediction = model.predict(data)

if prediction[0] == 1:
    print("🎉 Student will be PLACED")
else:
    print("❌ Student will NOT be placed")