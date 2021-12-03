let video, poseNet, pose;
let skeleton, brain, targetLabel, message;
let state = 'waiting';
let poseLabel = "";

function setup() {
  createCanvas(640,480);
  video = createCapture(VIDEO);
  video.hide();

  poseNet = ml5.poseNet(video, modelLoaded);
  poseNet.on('pose', gotPoses);
  message = createDiv('');

  options = {
    inputs: 34,
    outputs: 4,
    task: 'classification',
    debug: true
  }

  brain = ml5.neuralNetwork(options);
}

function keyPressed() {
  if (key == 't'){
    brain.normalizeData();
    brain.train({epochs: 50}, finished);
  } else {
    targetLabel = key;
    console.log(targetLabel);
    setTimeout(function() {
        state = 'Collecting';
        message.html('Started Collecting Data!');
        message.position(20, 500);
        message.style('font-size', '25px');
        console.log('Collecting data for pose:', targetLabel);
        setTimeout(function() {
          state = 'waiting';
          console.log('Stopped collecting data for pose:', targetLabel);
          message.html('Stopped Collecting Data!');
       }, 10000);
    }, 10000);
  }
}

function finished() {
  console.log('Model Trained!');
  console.log('Pose Estimation Ready!');
  classifyPose();
}

function classifyPose() {
  if (pose) {
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, gotResult);
  } else {
    setTimeout(classifyPose, 100);
  }
}

function gotResult(error, results) {  
  //console.log(results);
  //console.log(results[0].label);
  if (results[0].confidence > 0.75) {
    poseLabel = results[0].label.toUpperCase();
  }
  classifyPose();
}

function gotPoses(poses) {
  //console.log(poses); 
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
    if (state == 'Collecting') {
      let inputs = [];
      for (let i = 0; i < pose.keypoints.length; i++) {
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        inputs.push(x);
        inputs.push(y);
      }
      let target = [targetLabel];
      brain.addData(inputs, target);
    }
  }
}

function modelLoaded() {
  console.log('PoseNet Ready!');
}

function draw() {
  push();
  translate(video.width, 0);
  scale(-1, 1);
  image(video, 0, 0, video.width, video.height);

  if (pose) {
    for(let i = 0; i < skeleton.length; i++){
      let a = skeleton[i][0];
      let b = skeleton[i][1];

      strokeWeight(2);
      stroke(0);
      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }

    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0);
      stroke(255);
      ellipse(x, y, 16);
    }
  }
  pop();
  
  fill(255, 0, 255);
  noStroke();
  textSize(512);
  textAlign(CENTER, CENTER);
  text(poseLabel, width / 2, height / 2);
}
