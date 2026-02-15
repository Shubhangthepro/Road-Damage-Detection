RoadSight
Intelligent Road Damage Detection & Predictive Reporting System

RoadSight is an end-to-end AI-powered system that detects, classifies, verifies, and predicts road damage using citizen-uploaded images, GPS metadata, and weather analytics.

It combines deep learning, duplicate filtering, hotspot clustering, and predictive maintenance into a unified cloud-based platform for smarter infrastructure management.

🚀 Key Features

🔍 Dual-Stage AI Pipeline

YOLOv8 for damage detection

ResNet18 CNN for severity classification

Classifies roads as: Good, Satisfactory, Poor, Very Poor

🌦 Weather-Based Forecasting

Integrates rainfall, temperature, freeze-thaw data

Predicts deterioration risk

Computes a Road Health Index

📍 Crowdsourced GPS Reporting

Automatic geo-tagging

Citizen-powered data collection

🗺 Hotspot Mapping

Clusters repeated damage areas

Generates dynamic heatmaps

🔁 Duplicate & Fraud Filtering

Image hashing + GPS-time validation

Removes spam and redundant reports

📊 Live Admin Dashboard

Interactive map view

Status tracking (New, Scheduled, In Progress, Resolved)

Transparent citizen tracking

☁️ Cloud-Native Architecture

Flask backend

MongoDB Atlas

OpenStreetMap integration

Scalable & modular design

🧠 Problem Statement

Traditional road inspections are:

Manual and slow

Labor-intensive

Reactive instead of predictive

Prone to duplicate or false reports

Existing AI solutions focus only on pothole detection and lack:

Severity classification

Predictive analytics

Duplicate filtering

Integrated citizen dashboards

RoadSight solves this by unifying detection, validation, reporting, and forecasting into a single intelligent workflow.

🏗 Architecture Overview

User Upload (Mobile/Web)

Image + GPS + Timestamp

Stage 1: Detection

YOLOv8 identifies road defects

Stage 2: Severity Classification

ResNet18 classifies damage level

Duplicate Filtering

Image fingerprint + geo-radius check

Weather Forecast Integration

Degradation Risk Score

Road Health Index

Database Storage

MongoDB Atlas

Authority Alerts

Auto email/SMS for severe cases

Dashboard Visualization

Heatmaps

Live status tracking

📈 Results

🎯 97–98% overall accuracy

⚡ < 3 seconds per image processing

🔁 80% duplicate reports filtered

📍 Successful hotspot identification

🌧 Accurate prediction of pothole formation after heavy rain

💡 Technical Advantages

Reduces manual inspection costs

Enables predictive maintenance

Improves public transparency

Scales across cities without new hardware

Data-driven policy support

🔬 Methodology Summary

Dataset: Public road damage datasets (Kaggle + annotated data)

Object Detection: YOLOv8

Severity Classification: ResNet18

Backend: Flask

Database: MongoDB Atlas

Mapping: OpenStreetMap API

Forecasting: Weather-integrated regression model

🌍 Impact

RoadSight transforms road maintenance from reactive repair to predictive infrastructure intelligence, reducing costs, improving safety, and increasing citizen engagement.

📌 Future Enhancements

Real-time traffic integration

Contractor assignment automation

Government GIS API integration

Edge-device deployment
