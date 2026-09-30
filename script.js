/* Add or duplicate an entry below for each comparison; there is no row limit.
 * Replace prompt with the exact task prompt when available. Current RoboTwin
 * titles are formatted folder task identifiers; Franka prompts are user-provided.
 * Videos always render as Ground Truth | Ctrl-World | Token-World.
 */
const qualitativeVideoExperiments = [
  {
    "prompt": "Handover block",
    "videos": {
      "Ground Truth": "videos/qualitative/08_success_handover_block_robotwin_success_000431/gt.mp4",
      "Ctrl-World": "videos/qualitative/08_success_handover_block_robotwin_success_000431/ctrlworld.mp4",
      "Token-World": "videos/qualitative/08_success_handover_block_robotwin_success_000431/tokenworld.mp4"
    }
  },
  {
    "prompt": "Open laptop",
    "videos": {
      "Ground Truth": "videos/qualitative/12_success_open_laptop_robotwin_success_000841/gt.mp4",
      "Ctrl-World": "videos/qualitative/12_success_open_laptop_robotwin_success_000841/ctrlworld.mp4",
      "Token-World": "videos/qualitative/12_success_open_laptop_robotwin_success_000841/tokenworld.mp4"
    }
  },
  {
    "prompt": "Put object cabinet",
    "videos": {
      "Ground Truth": "videos/qualitative/191_failure_put_object_cabinet_robotwin_fail_001959/gt.mp4",
      "Ctrl-World": "videos/qualitative/191_failure_put_object_cabinet_robotwin_fail_001959/ctrlworld.mp4",
      "Token-World": "videos/qualitative/191_failure_put_object_cabinet_robotwin_fail_001959/tokenworld.mp4"
    }
  },
  {
    "prompt": "Stack blocks three",
    "videos": {
      "Ground Truth": "videos/qualitative/325_success_stack_blocks_three_robotwin_success_002235/gt.mp4",
      "Ctrl-World": "videos/qualitative/325_success_stack_blocks_three_robotwin_success_002235/ctrlworld.mp4",
      "Token-World": "videos/qualitative/325_success_stack_blocks_three_robotwin_success_002235/tokenworld.mp4"
    }
  },
  {
    "prompt": "Open microwave",
    "videos": {
      "Ground Truth": "videos/qualitative/434_success_open_microwave_robotwin_success_000890/gt.mp4",
      "Ctrl-World": "videos/qualitative/434_success_open_microwave_robotwin_success_000890/ctrlworld.mp4",
      "Token-World": "videos/qualitative/434_success_open_microwave_robotwin_success_000890/tokenworld.mp4"
    }
  },
  {
    "prompt": "Place can basket",
    "videos": {
      "Ground Truth": "videos/qualitative/484_success_place_can_basket_robotwin_success_001295/gt.mp4",
      "Ctrl-World": "videos/qualitative/484_success_place_can_basket_robotwin_success_001295/ctrlworld.mp4",
      "Token-World": "videos/qualitative/484_success_place_can_basket_robotwin_success_001295/tokenworld.mp4"
    }
  },
  {
    "prompt": "Pick the mcdonald on the table, hang it on the shelf.",
    "videos": {
      "Ground Truth": "videos/real/episode_000020/gt.mp4",
      "Ctrl-World": "videos/real/episode_000020/ctrlworld.mp4",
      "Token-World": "videos/real/episode_000020/tokenworld.mp4"
    }
  },
  {
    "prompt": "Pick the cup on the table, hang it on the shelf.",
    "videos": {
      "Ground Truth": "videos/real/episode_000189/gt.mp4",
      "Ctrl-World": "videos/real/episode_000189/ctrlworld.mp4",
      "Token-World": "videos/real/episode_000189/tokenworld.mp4"
    }
  },
  {
    "prompt": "Pick the chili on the table, place it in the drawer.",
    "videos": {
      "Ground Truth": "videos/real/episode_000304/gt.mp4",
      "Ctrl-World": "videos/real/episode_000304/ctrlworld.mp4",
      "Token-World": "videos/real/episode_000304/tokenworld.mp4"
    }
  },
  {
    "prompt": "Pick the jenga on the table, place it in the drawer.",
    "videos": {
      "Ground Truth": "videos/real/episode_000354/gt.mp4",
      "Ctrl-World": "videos/real/episode_000354/ctrlworld.mp4",
      "Token-World": "videos/real/episode_000354/tokenworld.mp4"
    }
  },
  {
    "prompt": "Pick the jenga on the left side, stack it on the other jenga",
    "videos": {
      "Ground Truth": "videos/real/episode_000461/gt.mp4",
      "Ctrl-World": "videos/real/episode_000461/ctrlworld.mp4",
      "Token-World": "videos/real/episode_000461/tokenworld.mp4"
    }
  },
  {
    "prompt": "Pick the yellow ring on the left side, stack it on the other ring.",
    "videos": {
      "Ground Truth": "videos/real/episode_000578/gt.mp4",
      "Ctrl-World": "videos/real/episode_000578/ctrlworld.mp4",
      "Token-World": "videos/real/episode_000578/tokenworld.mp4"
    }
  }
];

const videoComparisons = [];

function setupFigureFallbacks() {
  document.querySelectorAll(".figure-frame img").forEach((image) => {
    const frame = image.closest(".figure-frame");
    const markLoaded = () => frame.classList.add("is-loaded");
    image.addEventListener("load", markLoaded, { once: true });
    image.addEventListener("error", () => frame.classList.remove("is-loaded"), { once: true });
    if (image.complete && image.naturalWidth > 0) markLoaded();
  });
}

function createVideoCell(label, path) {
  const cell = document.createElement("div");
  cell.className = "video-cell";

  const labelElement = document.createElement("span");
  labelElement.className = "video-label";
  labelElement.textContent = label;

  const frame = document.createElement("div");
  frame.className = "video-frame";
  frame.dataset.videoPath = path;

  const video = document.createElement("video");
  video.muted = true;
  video.defaultMuted = true;
  video.playsInline = true;
  video.loop = true;
  video.preload = "metadata";
  video.setAttribute("muted", "");
  video.setAttribute("playsinline", "");
  video.setAttribute("loop", "");
  video.setAttribute("preload", "metadata");
  // Bypass cached copies of the original, unsupported MPEG-4 Part 2 files.
  video.src = path.startsWith("videos/real/") ? `${path}?v=h264` : path;
  video.setAttribute("aria-label", label);

  const placeholder = document.createElement("div");
  placeholder.className = "video-placeholder";
  placeholder.textContent = "Video placeholder";
  const pathElement = document.createElement("small");
  pathElement.textContent = path;
  placeholder.append(pathElement);

  video.addEventListener("loadedmetadata", () => {
    frame.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    frame.classList.add("is-loaded");
  });
  video.addEventListener("error", () => frame.classList.remove("is-loaded"));

  frame.append(video, placeholder);
  cell.append(labelElement, frame);
  return { cell, video };
}

function setupVideoComparison(experiment) {
  const wrapper = document.createElement("article");
  wrapper.className = "experiment";
  wrapper.dataset.experiment = experiment.prompt;

  const header = document.createElement("div");
  header.className = "experiment-header";

  const title = document.createElement("h3");
  title.className = "experiment-title";
  title.textContent = experiment.prompt;

  const controls = document.createElement("div");
  controls.className = "experiment-controls";
  controls.setAttribute("aria-label", `${experiment.prompt} controls`);

  const videoGrid = document.createElement("div");
  videoGrid.className = "video-grid";

  const videos = [];
  Object.entries(experiment.videos).forEach(([label, path]) => {
    const result = createVideoCell(label, path);
    videos.push(result.video);
    videoGrid.append(result.cell);
  });

  const availableVideos = () => videos.filter((video) => video.readyState > 0 && !video.error);
  let userInteracted = false;

  const syncToMaster = () => {
    const playable = availableVideos();
    if (!playable.length) return;
    const master = playable[0];
    playable.slice(1).forEach((video) => {
      if (Math.abs(video.currentTime - master.currentTime) > 0.08) {
        try { video.currentTime = master.currentTime; } catch (_) { /* no-op */ }
      }
    });
  };

  let syncTimer = null;
  const stopSyncTimer = () => {
    if (syncTimer !== null) window.clearInterval(syncTimer);
    syncTimer = null;
  };

  const startSyncTimer = () => {
    stopSyncTimer();
    syncTimer = window.setInterval(syncToMaster, 250);
  };

  const addButton = (label, action) => {
    const button = document.createElement("button");
    button.className = "control-button";
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", action);
    controls.append(button);
  };

  const playGroup = (restart = false) => {
    const playable = availableVideos();
    if (!playable.length) return;
    if (restart) {
      playable.forEach((video) => { video.currentTime = 0; });
    } else {
      syncToMaster();
    }
    // Start all three in the same turn. A pending play promise must not restart
    // the sync timer after the visitor has pressed Pause.
    startSyncTimer();
    playable.forEach((video) => {
      video.muted = true;
      video.play().catch(() => {
        // Browsers may restrict autoplay; the existing Play button can retry.
        if (playable.every((item) => item.paused)) stopSyncTimer();
      });
    });
  };

  addButton("Play", () => {
    userInteracted = true;
    playGroup();
  });

  addButton("Pause", () => {
    userInteracted = true;
    videos.forEach((video) => {
      video.autoplay = false;
      video.pause();
    });
    stopSyncTimer();
  });

  addButton("Replay", () => {
    userInteracted = true;
    playGroup(true);
  });

  videoComparisons.push({
    videos,
    autoplay() {
      // A visitor's controls take precedence over delayed startup.
      if (userInteracted) return;
      videos.forEach((video) => { video.autoplay = true; });
      playGroup(true);
    }
  });

  header.append(title, controls);
  wrapper.append(header, videoGrid);
  return wrapper;
}

function renderExperiments(targetId, experiments, renderer) {
  const target = document.getElementById(targetId);
  experiments.forEach((experiment, index) => target.append(renderer(experiment, index)));
}

function waitForVideoMetadata(video) {
  if (video.readyState > 0 || video.error) return Promise.resolve();
  return new Promise((resolve) => {
    const finish = () => {
      clearTimeout(timeout);
      video.removeEventListener("loadedmetadata", finish);
      video.removeEventListener("error", finish);
      resolve();
    };
    // Missing or slow assets must not hold up every demo indefinitely.
    const timeout = setTimeout(finish, 10000);
    video.addEventListener("loadedmetadata", finish);
    video.addEventListener("error", finish);
  });
}

async function startAllComparisons() {
  // One startup barrier for the whole page, then every demo starts together.
  await Promise.all(videoComparisons.flatMap(({ videos }) => videos.map(waitForVideoMetadata)));
  videoComparisons.forEach((comparison) => comparison.autoplay());
}

setupFigureFallbacks();
renderExperiments("qualitative-video-results", qualitativeVideoExperiments, setupVideoComparison);
startAllComparisons();
