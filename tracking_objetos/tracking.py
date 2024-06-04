import cv2

# algoritmo de tracking KCF, rapido de exec mas nao indicado para videos rapidos
#rastreador = cv2.TrackerKCF_create()
# algoritmo de tracking CSRT, mais lento mas mais preciso
rastreador = cv2.TrackerCSRT_create()

#video = cv2.VideoCapture('tracking_objetos\\videos\\race.mp4')
video = cv2.VideoCapture('tracking_objetos\\videos\\street.mp4')
ok, frame = video.read()

bbox = cv2.selectROI(frame) # regiao de interesse
print(bbox)

ok = rastreador.init(frame, bbox)
# print(ok)

while True:
    ok, frame = video.read()
    # print(ok)
    if not ok:
        break
    
    ok, bbox = rastreador.update(frame)
    print(bbox)

    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2, 1)
    else:
        cv2.putText(frame, 'Perdeu o objeto', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
