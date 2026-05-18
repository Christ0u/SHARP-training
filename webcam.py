from ultralytics import YOLO
import cv2

# Charger ton modèle entraîné
model = YOLO("runs/detect/runs/train/weights/best.pt")

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prédiction
    results = model(frame, conf=0.5)

    # Dessiner les résultats
    annotated_frame = results[0].plot()

    # Compter les objets détectés (doigts)
    nb_objets = len(results[0].boxes)

    # Afficher le nombre
    cv2.putText(
        annotated_frame,
        f"Doigts: {nb_objets}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Affichage
    cv2.imshow("Detection doigts", annotated_frame)

    # Quitter avec Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()