<template>
  <div class="container">
    <div class="row">
      <div class="col-12 text-center">
        <h1 class="text-uppercase">Mention classifier</h1>
        <h4 class="font-weight-light">
          <small>Advansed information extraction</small>
        </h4>
      </div>
    </div>

    <div class="row pt-4 pb-4">
      <div class="offset-1 col-4 text-center">
        <img width="300px" src="@/assets/dots.svg">
      </div>
      <div class="col-7">
        <h4>
          The next step after
          <abbr title="Named Entity Recognition" class="initialism">NER</abbr>
        </h4>
        <p>
          Mention classification lays between Named Entity Recognition and Entity Linking.
          It provides much more detailed information about an object than NER does.
          At the same time, it does not require storing any knowledge graph with known objects.
          The Classifier able to work with absolutely new objects never appeared in the train set.
        </p>
      </div>
    </div>

    <div class="row pt-4 pb-4">
      <div class="col-7 offset-1 d-flex align-items-center">
        <div>
          <h4>Possible applications</h4>
          <p>
            Large scale knowledge extraction from news feeds or any web source for specific mentions.
            Better slot filling provides additional information for chatbots and question answering systems.
          </p>
        </div>
      </div>
      <div class="col-4 text-center">
        <img width="200px" src="@/assets/search.svg">
      </div>
    </div>


    <div class="row">
      <div class="col-12 text-center">
        <img src="@/assets/nn.png">
      </div>
    </div>


    <div class="row">
      <div class="col-12 text-center">
        <h2 class="text-uppercase">Features</h2>
      </div>
    </div>

    <div class="row pt-4">
      <div class="offset-2 col-1 text-center">
        <h1>
          <font-awesome-icon icon="project-diagram"/>
        </h1>
      </div>
      <div class="col-7">
        The neural net is able to simulteniously look at up to five mentions.
        It combines evidences more efficient than it could be achived by averaging individual predictions.
      </div>
    </div>

    <div class="row pt-4 pb-4">
      <div class="offset-2 col-7 text-right">
        <span class="align-middle">
          Model known
          <b>more than 200</b> general people categories.
          It's also possible to add specific categories without complete model retraining.
        </span>
      </div>
      <div class="col-1 text-center">
        <img width="70px" src="@/assets/many_classes.svg">
      </div>
      <div class="col-2"></div>
    </div>

    <div class="text-secondary">
      <div class="row pt-4">
        <div class="col-12 text-center">
          <h2 class="font-weight-light">
            <small>Comming soon</small>
          </h2>
        </div>
      </div>

      <div class="row pt-3">
        <div class="offset-2 col-1 d-flex align-items-center justify-content-center">
          <h1>
            <font-awesome-icon icon="language"/>
          </h1>
        </div>
        <div class="col-3">
          <h5>Support for other languages</h5>
          <p>
            <font-awesome-icon icon="angle-right"/>&nbsp;Русский
            <font-awesome-icon icon="angle-right"/>&nbsp;Deutsch
            <font-awesome-icon icon="angle-right"/>&nbsp;Français
          </p>
        </div>
        <!-- </div>

        <div class="row pt-2">-->
        <div class="col-1 d-flex align-items-center justify-content-center">
          <h1>
            <font-awesome-icon icon="expand-arrows-alt"/>
          </h1>
        </div>
        <div class="col-4">
          <h5>Support for other entities</h5>
          <p>
            <font-awesome-icon icon="building"/>&nbsp;Organizations &nbsp;
            <font-awesome-icon icon="calendar-check"/>&nbsp;Events &nbsp;
            <font-awesome-icon icon="pizza-slice"/>&nbsp;Products &nbsp;
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
// @ is an alias to /src
import { library } from "@fortawesome/fontawesome-svg-core";
import {
  faProjectDiagram,
  faBraille,
  faLanguage,
  faAngleRight,
  faCalendarCheck,
  faBuilding,
  faPizzaSlice,
  faExpandArrowsAlt
} from "@fortawesome/free-solid-svg-icons";

library.add(faProjectDiagram);
library.add(faBraille);
library.add(faLanguage);
library.add(faAngleRight);
library.add(faCalendarCheck);
library.add(faBuilding);
library.add(faPizzaSlice);
library.add(faExpandArrowsAlt);

export default {
  name: "home"
};
</script>

<style>
</style>
