package org.reactnative.facedetector;

import java.util.List;
import android.util.Log;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableNativeArray;
import com.google.firebase.ml.vision.common.FirebaseVisionPoint;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceLandmark;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceContour;

public class FaceDetectorUtils {
    private static final String[] landmarkNames = {
            "bottomMouthPosition", "leftCheekPosition", "leftEarPosition",
            "leftEyePosition", "leftMouthPosition", "noseBasePosition", "rightCheekPosition",
            "rightEarPosition", "rightEyePosition", "rightMouthPosition"
    };
    private static final String[] contourNames = {
            "face",
            "leftEye", "leftEyebrowBottom", "leftEyebrowTop",
            "lowerLipBottom", "lowerLipTop",
            "noseBottom", "noseBridge",
            "rightEye", "rightEyebrowBottom", "rightEyebrowTop",
            "upperLipBottom", "upperLipTop",
    };

    public static WritableMap serializeFace(FirebaseVisionFace face) {
        return serializeFace(face, 1, 1, 0, 0, 0, 0);
    }

    public static WritableMap serializeFace(FirebaseVisionFace face, double scaleX, double scaleY, int width, int height, int paddingLeft, int paddingTop) {
        WritableMap encodedFace = Arguments.createMap();

        int id = 0;
        // If face tracking was enabled:
        if (face.getTrackingId() != FirebaseVisionFace.INVALID_ID) {
            id = face.getTrackingId();
        }


        encodedFace.putInt("faceID", id);
        encodedFace.putDouble("rollAngle", face.getHeadEulerAngleZ());
        encodedFace.putDouble("yawAngle", face.getHeadEulerAngleY());

        // If classification was enabled:
        if (face.getSmilingProbability() != FirebaseVisionFace.UNCOMPUTED_PROBABILITY) {
            encodedFace.putDouble("smilingProbability", face.getSmilingProbability());
        }
        if (face.getLeftEyeOpenProbability() != FirebaseVisionFace.UNCOMPUTED_PROBABILITY) {
            encodedFace.putDouble("leftEyeOpenProbability", face.getLeftEyeOpenProbability());
        }
        if (face.getRightEyeOpenProbability() != FirebaseVisionFace.UNCOMPUTED_PROBABILITY) {
            encodedFace.putDouble("rightEyeOpenProbability", face.getRightEyeOpenProbability());
        }
        int[] landmarks = {
                FirebaseVisionFaceLandmark.MOUTH_BOTTOM,
                FirebaseVisionFaceLandmark.LEFT_CHEEK,
                FirebaseVisionFaceLandmark.LEFT_EAR,
                FirebaseVisionFaceLandmark.LEFT_EYE,
                FirebaseVisionFaceLandmark.MOUTH_LEFT,
                FirebaseVisionFaceLandmark.NOSE_BASE,
                FirebaseVisionFaceLandmark.RIGHT_CHEEK,
                FirebaseVisionFaceLandmark.RIGHT_EAR,
                FirebaseVisionFaceLandmark.RIGHT_EYE,
                FirebaseVisionFaceLandmark.MOUTH_RIGHT
        };

        for (int i = 0; i < landmarks.length; ++i) {
            FirebaseVisionFaceLandmark landmark = face.getLandmark(landmarks[i]);
            if (landmark != null) {
                encodedFace.putMap(landmarkNames[i], mapFromPoint(landmark.getPosition(), scaleX, scaleY, width, height, paddingLeft, paddingTop));
            }
        }

        int[] contours = {
                FirebaseVisionFaceContour.FACE,
                FirebaseVisionFaceContour.LEFT_EYE,
                FirebaseVisionFaceContour.LEFT_EYEBROW_BOTTOM,
                FirebaseVisionFaceContour.LEFT_EYEBROW_TOP,
                FirebaseVisionFaceContour.LOWER_LIP_BOTTOM,
                FirebaseVisionFaceContour.LOWER_LIP_TOP,
                FirebaseVisionFaceContour.NOSE_BOTTOM,
                FirebaseVisionFaceContour.NOSE_BRIDGE,
                FirebaseVisionFaceContour.RIGHT_EYE,
                FirebaseVisionFaceContour.RIGHT_EYEBROW_BOTTOM,
                FirebaseVisionFaceContour.RIGHT_EYEBROW_TOP,
                FirebaseVisionFaceContour.UPPER_LIP_BOTTOM,
                FirebaseVisionFaceContour.UPPER_LIP_TOP,
        };

        for (int i = 0; i < contours.length; ++i) {
            FirebaseVisionFaceContour contour = face.getContour(contours[i]);
            if (contour != null) {
                WritableArray pointsArray = new WritableNativeArray();
                List<FirebaseVisionPoint> pointList = contour.getPoints();
                for (int j = 0; j < pointList.size(); j++) {
                    pointsArray.pushMap(mapFromPoint(pointList.get(j), scaleX, scaleY, width, height, 0, 0));
                }
                encodedFace.putArray(contourNames[i], pointsArray);
            }
        }

        WritableMap origin = Arguments.createMap();
        Float x = face.getBoundingBox().exactCenterX() - (face.getBoundingBox().width() / 2);
        Float y = face.getBoundingBox().exactCenterY() - (face.getBoundingBox().height() / 2);
        if (face.getBoundingBox().exactCenterX() < width / 2) {
            x = x + paddingLeft / 2;
        } else if (face.getBoundingBox().exactCenterX() > width / 2) {
            x = x - paddingLeft / 2;
        }

        if (face.getBoundingBox().exactCenterY() < height / 2) {
            y = y + paddingTop / 2;
        } else if (face.getBoundingBox().exactCenterY() > height / 2) {
            y = y - paddingTop / 2;
        }

        origin.putDouble("x", x * scaleX);
        origin.putDouble("y", y * scaleY);

        WritableMap size = Arguments.createMap();
        size.putDouble("width", face.getBoundingBox().width() * scaleX);
        size.putDouble("height", face.getBoundingBox().height() * scaleY);

        WritableMap bounds = Arguments.createMap();
        bounds.putMap("origin", origin);
        bounds.putMap("size", size);

        encodedFace.putMap("bounds", bounds);

        return encodedFace;
    }

    public static WritableMap rotateFaceX(WritableMap face, int sourceWidth, double scaleX) {
        ReadableMap faceBounds = face.getMap("bounds");

        ReadableMap oldOrigin = faceBounds.getMap("origin");
        WritableMap mirroredOrigin = positionMirroredHorizontally(oldOrigin, sourceWidth, scaleX);

        double translateX = -faceBounds.getMap("size").getDouble("width");
        WritableMap translatedMirroredOrigin = positionTranslatedHorizontally(mirroredOrigin, translateX);

        WritableMap newBounds = Arguments.createMap();
        newBounds.merge(faceBounds);
        newBounds.putMap("origin", translatedMirroredOrigin);

        for (String landmarkName : landmarkNames) {
            ReadableMap landmark = face.hasKey(landmarkName) ? face.getMap(landmarkName) : null;
            if (landmark != null) {
                WritableMap mirroredPosition = positionMirroredHorizontally(landmark, sourceWidth, scaleX);
                face.putMap(landmarkName, mirroredPosition);
            }
        }

        face.putMap("bounds", newBounds);

        return face;
    }

    public static WritableMap changeAnglesDirection(WritableMap face) {
        face.putDouble("rollAngle", (-face.getDouble("rollAngle") + 360) % 360);
        face.putDouble("yawAngle", (-face.getDouble("yawAngle") + 360) % 360);
        return face;
    }

    public static WritableMap mapFromPoint(FirebaseVisionPoint point, double scaleX, double scaleY, int width, int height, int paddingLeft, int paddingTop) {
        WritableMap map = Arguments.createMap();
        Float x = point.getX();
        Float y = point.getY();
        if (point.getX() < width / 2) {
            x = (x + paddingLeft / 2);
        } else if (point.getX() > width / 2) {
            x = (x - paddingLeft / 2);
        }

        if (point.getY() < height / 2) {
            y = (y + paddingTop / 2);
        } else if (point.getY() > height / 2) {
            y = (y - paddingTop / 2);
        }
        map.putDouble("x", x * scaleX);
        map.putDouble("y", y * scaleY);
        return map;
    }

    public static WritableMap positionTranslatedHorizontally(ReadableMap position, double translateX) {
        WritableMap newPosition = Arguments.createMap();
        newPosition.merge(position);
        newPosition.putDouble("x", position.getDouble("x") + translateX);
        return newPosition;
    }

    public static WritableMap positionMirroredHorizontally(ReadableMap position, int containerWidth, double scaleX) {
        WritableMap newPosition = Arguments.createMap();
        newPosition.merge(position);
        newPosition.putDouble("x", valueMirroredHorizontally(position.getDouble("x"), containerWidth, scaleX));
        return newPosition;
    }

    public static double valueMirroredHorizontally(double elementX, int containerWidth, double scaleX) {
        double originalX = elementX / scaleX;
        double mirroredX = containerWidth - originalX;
        return mirroredX * scaleX;
    }
}
