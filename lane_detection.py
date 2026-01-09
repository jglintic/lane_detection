import cv2
import numpy as np
import glob

def calibration():
    # number of interior angles
    rows = 6
    cols = 9

    # Set the termination criteria for the corner sub-pixel algorithm
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0). They are the same for all images
    objectPoints = np.zeros((rows * cols, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    # dots arrays
    objpoints = []  # 3D points in real space
    imgpoints = []  # 2D points in image

    # load images
    images = glob.glob("camera_cal/*.jpg")

    if len(images) == 0:
        raise RuntimeError("No figures found in calibration_images folder!")

    # image processing
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

        if ret:
            objpoints.append(objectPoints)

            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners)

            # display images
            cv2.drawChessboardCorners(img, (rows, cols), corners, ret)

            #cv2.imshow("Detected corners", img)
            #cv2.waitKey(150)

        else:
            print(f"Angles not found in the image: {fname}")

    cv2.destroyAllWindows()

    # Calibrate the camera and save the results
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # print results
    print("\n=== Calibration done ===")
    print("Camera matrix:\n", camera_matrix)
    print("\nDistortion coefficients:\n", dist_coeffs)

    # save in npz file
    np.savez('camera_params.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)

    print("\nCamara parameters saved in 'camera_params.npz'")

def drawLane(img, lanePoints):
    # minimum 3 points to draw a lane
    if len(lanePoints) < 3:
        return img

    xCoordinates = np.array([p[0] for p in lanePoints])
    yCoordinates = np.array([p[1] for p in lanePoints])

    fit = np.polyfit(xCoordinates, yCoordinates, 2)
    a, b, c = fit
    height, width = img.shape[:2]
    lanePoints2 = []
    for x in range(min(xCoordinates), max(xCoordinates) + 1):
        y = int(a * x**2 + b * x + c)
        if 0 <= y < height:
            lanePoints2.append((x, y))

    lanePoints2 = np.array([lanePoints2], dtype=np.int32)
    # draw lanes in the image
    cv2.polylines(img, lanePoints2, isClosed=False, color=255, thickness=30)

def main():
    # camera calibration
    calibration()
    calib = np.load('camera_params.npz')
    camera_matrix = calib['camera_matrix']
    dist_coeffs = calib['dist_coeffs']

    # undistorted chess image
    originalChessImg = cv2.imread('camera_cal/calibration3.jpg')
    chessImgH, chessImgW = originalChessImg.shape[:2]
    newCameraMtxChess, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (chessImgW, chessImgH), 1, (chessImgW, chessImgH))
    undistortedChessImg = cv2.undistort(originalChessImg, camera_matrix, dist_coeffs, None, newCameraMtxChess)
    cv2.imshow('Undistorted chess image', np.hstack((undistortedChessImg, originalChessImg)))

    # video processing
    cap = cv2.VideoCapture('test_videos/project_video02.mp4')
    outputVideoPath = 'video_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    frameW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_output = cv2.VideoWriter(outputVideoPath, fourcc, fps, (frameW, frameH))

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Video closed")
            break

        imgH, imgW = img.shape[:2]
        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (imgW, imgH), 1, (imgW, imgH))
        undistortedImg = cv2.undistort(img, camera_matrix, dist_coeffs, None, newCameraMtx)
        cv2.imshow('Undistorted orignal image', np.hstack((img, undistortedImg)))

        srcPoints = np.float32([
            [imgW * 19 / 100, imgH], # Bottom-left corner
            [imgW, imgH],  # Bottom-right corner
            [imgW * 61 / 100, imgH * 64 / 100],  # Top-right corner
            [imgW * 50 / 100, imgH * 64 / 100]  # Top-left corner
        ])

        dstPoints = np.float32([
            [0, imgH],  # Bottom-left corner
            [imgW, imgH],  # Bottom-right corner
            [imgW, 0],  # Top-right corner
            [0, 0]  # Top-left corner
        ])

        mask = np.ones((imgH, imgW), dtype=np.uint8) * 255
        srcPointsInt = srcPoints.astype(np.int32)
        cv2.fillPoly(mask, [srcPointsInt], (0, 0, 0))
        maskedImage = cv2.bitwise_and(undistortedImg, undistortedImg, mask=mask)

        transformMatrix = cv2.getPerspectiveTransform(srcPoints, dstPoints)
        warpedImage = cv2.warpPerspective(undistortedImg, transformMatrix, (imgW, imgH))
        warpedImage2 = warpedImage.copy()
        cv2.imshow('Perspective transformed image', np.hstack((undistortedImg, warpedImage)))

        srcPointsInt = srcPoints.astype(np.int32)
        cv2.fillPoly(mask, [srcPointsInt], (0, 0, 0))

        gray = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray, (7, 7), 0)
        ret, thrash = cv2.threshold(gaussian, 135, 150, cv2.THRESH_BINARY)
        canny = cv2.Canny(thrash, 120, 240)
        cv2.imshow("Binary Image", canny)

        lines = cv2.HoughLinesP(canny, 1, np.pi / 180, threshold=25, minLineLength=200, maxLineGap=200)

        rightLane = []
        leftLane = []
        midPoint = imgW // 2

        if lines is not None:
            for points in lines:
                x1, y1, x2, y2 = points[0]
                p1, p2 = (x1, y1), (x2, y2)
                cv2.line(warpedImage2, p1, p2, 255)

                if (x1 > midPoint):
                    rightLane.append(p1)
                    rightLane.append(p2)
                else:
                    leftLane.append(p1)
                    leftLane.append(p2)

        drawLane(warpedImage, rightLane)
        drawLane(warpedImage, leftLane)

        inverseMatrix = cv2.getPerspectiveTransform(dstPoints, srcPoints)
        reconstructedImage = cv2.warpPerspective(warpedImage, inverseMatrix, (imgW, imgH))
        finalImage = cv2.bitwise_or(maskedImage, reconstructedImage)
        cv2.imshow('Lane detection', np.hstack((undistortedImg, finalImage)))
        video_output.write(finalImage)

        if cv2.waitKey(60) & 0xFF == 27: break
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()