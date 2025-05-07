
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def load_and_preprocess_map(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    return binary

def extract_skeleton(binary):
    skeleton = cv2.ximgproc.thinning(binary)  # Requires ximgproc module
    return skeleton

def get_walkable_points(skeleton):
    points = np.column_stack(np.where(skeleton > 0))
    return [tuple(p) for p in points]

def build_graph(points, max_dist=10):
    G = nx.Graph()
    for i, p1 in enumerate(points):
        G.add_node(p1)
        for j in range(i+1, len(points)):
            p2 = points[j]
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist < max_dist:
                G.add_edge(p1, p2, weight=dist)
    return G

def click_location(image, title="Click Location"):
    coords = []
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.append((y, x))
            cv2.destroyAllWindows()

    cv2.imshow(title, image)
    cv2.setMouseCallback(title, on_click)
    cv2.waitKey(0)
    return coords[0] if coords else None

def find_closest_node(G, point):
    return min(G.nodes, key=lambda n: np.linalg.norm(np.array(n) - np.array(point)))

def draw_path(image, path):
    for i in range(len(path)-1):
        cv2.line(image, (path[i][1], path[i][0]), (path[i+1][1], path[i+1][0]), (0, 0, 255), 2)
    cv2.imshow("Path", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = "map.png"  # Put your map image here
    binary = load_and_preprocess_map(image_path)
    skeleton = extract_skeleton(binary)
    points = get_walkable_points(skeleton)
    graph = build_graph(points)

    original_img = cv2.imread(image_path)
    print("Click your current location...")
    current = click_location(original_img.copy(), "Current Location")
    print("Click your destination...")
    dest = click_location(original_img.copy(), "Destination")

    start_node = find_closest_node(graph, current)
    end_node = find_closest_node(graph, dest)
    path = nx.shortest_path(graph, start_node, end_node, weight='weight')
    
    draw_path(original_img, path)

if __name__ == "__main__":
    main()



import pyttsx3
import speech_recognition as sr
import json
import os

feedback_file = "feedback.json"
engine = pyttsx3.init()

def speak(text):
    print("[Assistant]:", text)
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except:
        speak("Sorry, I didn't catch that.")
        return None

def get_destination_from_user():
    speak("Where would you like to go?")
    return listen()

def save_feedback(start, end, path):
    data = {}
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            data = json.load(f)
    key = f"{start}_{end}"
    if key in data:
        data[key]["count"] += 1
    else:
        data[key] = {"path": path, "count": 1}
    with open(feedback_file, "w") as f:
        json.dump(data, f, indent=2)

def suggest_known_paths(start, end):
    if not os.path.exists(feedback_file):
        return None
    with open(feedback_file, "r") as f:
        data = json.load(f)
    key = f"{start}_{end}"
    if key in data:
        speak("I remember a preferred path. Using that.")
        return data[key]["path"]
    return None
