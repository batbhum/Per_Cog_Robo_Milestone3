# Perception of Cognitive Robots: Term project
**Date:** January 6, 2026

## 1 Overview
The goal of this project is to demonstrate mastery of the concepts of robotic perception, including image processing, simultaneous localization and mapping, anchoring, and simulation methodologies. This mastery will be demonstrated through the design and implementation of several simulations and associated reports, describing this process.

Simulation can be performed on WeBots, Gazebo, ARGOS, or any other physics-enabled robot simulator of choice.

The final simulation must encompass:
A single robot traversing a complex world, finding a path to a goal and avoiding obstacles.

* **The world consists of:**
    * A square environment, surrounded by walls on all sides, with several walls inside the square.
    * At least two moving objects that traverse the world obeying the laws of physics, assuming perfectly elastic collisions.
    * A goal.
* **The robot is equipped with:**
    * A single camera that displays the world in front of it
    * A range-sensing system (ultrasound, LiDar, etc.) that allows it to detect the distance to a sensed obstacle.

At each submission milestone, students will provide an updated simulation where robot control and coordination is improved, using the concepts learned in the course so far. The report at each milestone must describe the mathematics, and corresponding description of software implementation, of those concepts in the simulation.

Students may use any programming language of choice, but no libraries for image processing, computer vision, SLAM, or the likes (basic libraries for mathematical operations are allowed).

## 2 Assessment
Assessment will prioritize students' demonstration of mastery over the course's learning objectives, rather than the sophistication of the prototype. However, it is a necessary pre-condition that the prototype works.

Each assessment milestone accounts for 20% of the final grade.

## 3 Timeline
There are 3 milestones throughout the term. At each milestone, you must upload the report thus far, and demonstrate current working simulation, up until the current level of development.

1.  **6 of February:** Computer vision. Robot must be able to distinguish static from dynamic obstacles, and identify the goal.
2.  **6 of March:** SLAM. Robot must be able to construct a map of the world and keep track of its own position.
3.  **3 of April:** Anchoring and planning. Robot must be able to predict where dynamic objects are and plan a path towards the goal that avoids all obstacles.
