(function () {
  "use strict";

  function initialiseBirdMap() {
    const mapElement = document.getElementById("bird-map");
    const dataElement = document.getElementById("bird-data");

    if (!mapElement || !dataElement) {
      return;
    }

    if (typeof window.L === "undefined") {
      showMapError(mapElement, "The map library could not be loaded. The complete bird list is available below.");
      return;
    }

    let species;

    try {
      species = JSON.parse(dataElement.textContent);
    } catch (error) {
      showMapError(mapElement, "The observation data could not be loaded. The complete bird list is available below.");
      return;
    }

    const locations = groupObservationsByLocation(species);

    if (locations.length === 0) {
      showMapError(mapElement, "No mapped observations are available yet.");
      return;
    }

    mapElement.replaceChildren();

    const map = window.L.map(mapElement, {
      scrollWheelZoom: false
    });

    window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }).addTo(map);

    const markerLayer = typeof window.L.markerClusterGroup === "function"
      ? window.L.markerClusterGroup({
          maxClusterRadius: 35,
          showCoverageOnHover: false
        })
      : window.L.layerGroup();

    const bounds = [];

    locations.forEach(function (location) {
      const position = [location.latitude, location.longitude];
      const marker = window.L.marker(position, {
        alt: "Bird observations near " + location.name,
        title: location.name
      });

      marker.bindPopup(buildPopup(location), {
        maxHeight: 320,
        maxWidth: 310
      });
      markerLayer.addLayer(marker);
      bounds.push(position);
    });

    markerLayer.addTo(map);
    map.fitBounds(bounds, {
      maxZoom: 6,
      padding: [30, 30]
    });
  }

  function groupObservationsByLocation(species) {
    const locations = new Map();

    species.forEach(function (bird) {
      (bird.observations || []).forEach(function (observation) {
        const latitude = Number(observation.latitude);
        const longitude = Number(observation.longitude);

        if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
          return;
        }

        // Shared representative coordinates deliberately produce one marker,
        // even when records use slightly different names for the same region.
        const key = [latitude, longitude].join("|");

        if (!locations.has(key)) {
          locations.set(key, {
            name: observation.location,
            latitude: latitude,
            longitude: longitude,
            accuracy: observation.coordinate_accuracy || "unknown",
            birds: []
          });
        }

        locations.get(key).birds.push({
          id: bird.id,
          commonName: bird.common_name,
          scientificName: bird.scientific_name,
          subspecies: observation.subspecies || null
        });
      });
    });

    return Array.from(locations.values()).map(function (location) {
      location.birds.sort(function (first, second) {
        return first.commonName.localeCompare(second.commonName);
      });
      return location;
    });
  }

  function buildPopup(location) {
    const popup = document.createElement("div");
    popup.className = "bird-popup";

    const heading = document.createElement("strong");
    heading.textContent = location.name;
    popup.appendChild(heading);

    const list = document.createElement("ul");

    location.birds.forEach(function (bird) {
      const item = document.createElement("li");
      const link = document.createElement("a");
      const scientificName = document.createElement("em");

      link.href = "#" + encodeURIComponent(bird.id);
      link.textContent = bird.commonName;
      scientificName.textContent = bird.scientificName;

      item.appendChild(link);
      item.appendChild(document.createTextNode(" — "));
      item.appendChild(scientificName);

      if (bird.subspecies) {
        const subspecies = document.createElement("small");
        const subspeciesName = document.createElement("em");

        subspecies.appendChild(document.createElement("br"));
        subspecies.appendChild(document.createTextNode("ssp. " + bird.subspecies.name + " — "));
        subspeciesName.textContent = bird.subspecies.scientific_name;
        subspecies.appendChild(subspeciesName);
        item.appendChild(subspecies);
      }

      list.appendChild(item);
    });

    popup.appendChild(list);

    if (location.accuracy !== "exact") {
      const accuracy = document.createElement("small");
      accuracy.className = "bird-popup-accuracy";
      accuracy.textContent = "Approximate observation area (" + location.accuracy + ").";
      popup.appendChild(accuracy);
    }

    return popup;
  }

  function showMapError(mapElement, message) {
    mapElement.classList.add("bird-map--unavailable");
    mapElement.replaceChildren();

    const status = document.createElement("p");
    status.className = "bird-map-status";
    status.textContent = message;
    mapElement.appendChild(status);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialiseBirdMap);
  } else {
    initialiseBirdMap();
  }
}());
