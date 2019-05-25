<template>
<div>
<div class="lead-screen">
  <div class="container">
    <div class="row">
      <div class="col-12 text-center">
        <i></i>
        <h1 class="text-uppercase">Mention classifier</h1>
        <h4 class="font-weight-light mb-4">
          <small>{{$t('home.subtitle')}}</small>
        </h4>
        
        <router-link to="/demo"><button class="btn btn-primary mr-2">{{ $t('home.demo') }}</button></router-link>
        
        <a href='https://t.me/neural_network_engineering' target="blank">
        <button class="btn btn-primary">{{$t('home.subscribe')}} <font-awesome-icon :icon="['fab', 'telegram-plane']"/> </button>
        </a>
      </div>
    </div>
</div>
</div>
 
<div>
 <div class="container">
    <div class="row pt-5 pb-5">
      <div class="col-12 text-center">
        <img src="@/assets/example.svg" class="nn-image"><br/>
      </div>
    </div>

    <div class="row pt-5 pb-5">
      <div class="col-12 col-lg-6 text-center pt-4">
        <img width="300px" src="@/assets/dots.svg">
      </div>
      <div class="col-12 col-lg-6 pt-4 pt-lg-0">
        <h4 class="mb-3" v-html="$t('home.ner_title')">
          
        </h4>
        <p class="info-text" v-html="$t('home.ner_text')">
         
        </p>
      </div>
    </div>

    <div class="row pt-0 pt-lg-5 pb-5">
      <div class="col-12 col-lg-7 offset-0 offset-lg-1 d-flex align-items-center">
        <div>
          <h4 class="mb-3" v-html="$t('home.applications_title')"></h4>
          <p class="info-text" v-html="$t('home.applications_text')"></p>
        </div>
      </div>
      <div class="col-4 d-none d-lg-block text-center">
        <img width="200px" src="@/assets/search.svg">
      </div>
    </div>
    <div class="row pt-5 pb-5">
      <div class="col-12 text-center">
        <img src="@/assets/nn.png" class="nn-image"><br/>
        <small class="info-text" v-html="$t('home.nn_arch')"></small>
      </div>
    </div>
 </div>
</div>

<div class="second-screen">
 <div class="container">
    


    <div class="row">
      <div class="col-12 text-center">
        <h2 class="text-uppercase" v-html="$t('home.features')"></h2>
      </div>
    </div>

    <div class="row pt-4">
      <div class="offset-0 offset-lg-2 col-3 col-lg-1 text-center">
        <h1>
          <font-awesome-icon icon="project-diagram"/>
        </h1>
      </div>
      <div class="col-9 col-lg-7" v-html="$t('home.feature_one')"></div>
    </div>

    <div class="row pt-4 pb-4">
      <div class="offset-0 offset-lg-2 col-9 col-lg-7 text-right">
        <span class="align-middle" v-html="$t('home.feature_two')"></span>
      </div>
      <div class="col-3 col-lg-1 text-center">
        <img width="70px" src="@/assets/many_classes.svg">
      </div>
      <div class="col-2"></div>
    </div>

    <div class="text-secondary">
      <div class="row pt-4">
        <div class="col-12 text-center">
          <h2 class="font-weight-light">
            <small v-html="$t('home.coming_soon')"></small>
          </h2>
        </div>
      </div>

      <div class="row pt-3">
        <div class="offset-0 offset-lg-2 col-3 col-lg-1 d-flex align-items-center justify-content-center">
          <h1>
            <font-awesome-icon icon="language"/>
          </h1>
        </div>
        <div class="col-9 col-lg-3">
          <h5 v-html="$t('home.lang_support')"></h5>
          <p>
            <font-awesome-icon icon="angle-right"/>&nbsp;Русский
            <font-awesome-icon icon="angle-right"/>&nbsp;Deutsch
            <font-awesome-icon icon="angle-right"/>&nbsp;Français
          </p>
        </div>
        <!-- </div>

        <div class="row pt-2">-->
        <div class="col-3 col-lg-1 d-flex align-items-center justify-content-center">
          <h1>
            <font-awesome-icon icon="expand-arrows-alt"/>
          </h1>
        </div>
        <div class="col-9 col-lg-4">
          <h5 v-html="$t('home.other_entities')"></h5>
          <p>
            <font-awesome-icon icon="building"/>&nbsp;{{$t('home.organisations')}} &nbsp;
            <font-awesome-icon icon="calendar-check"/>&nbsp;{{$t('home.events')}} &nbsp;
            <font-awesome-icon icon="pizza-slice"/>&nbsp;{{$t('home.products')}} &nbsp;
          </p>
        </div>
      </div>
    </div>
  </div>
</div>
</div>
</template>

<script>
// @ is an alias to /src
import { library } from "@fortawesome/fontawesome-svg-core";

import { faTelegramPlane } from '@fortawesome/free-brands-svg-icons'

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
library.add(faTelegramPlane);

import LocaleSwitcher from "../components/locale_switcher";

export default {
  name: "home",
  components: {
    LocaleSwitcher
  }
};
</script>

<style>
  .lead-screen {
    display: flex;
    align-items: center;
    height: 500px;
    background: url(~@/assets/free_background_6.png) no-repeat;
    background-size: cover;
  }

  .nn-image {
    max-width: 100%;
  }

  .lead-screen i {
    font-size: 30px;
color: #fff;
display: block;
width: 40px;
height: 40px;
background: #fff;
border-radius: 50%;
margin: 20px auto 30px;
  }

   .lead-screen h1 {
      color: wheat;
      font-size: 60px;
      font-weight: 600;
      letter-spacing: 5px;
    }

    .lead-screen h4 {
      color: white;
      font-size: 32px;
      font-style: italic;
      font-family: Georgia, 'Times New Roman', Times, serif;
    }

    .lead-screen .btn-primary {
      background-color: #0a6995;
      border-color: #0a6995;
    }
    .lead-screen .btn-primary:hover {
      background-color: #1a7ba8;
      border-color: #1a7ba8;
    }
    
    .info-text {
      color: #747474;
    }

    .second-screen {
      background: #dddddd;
      clip-path: polygon(0 0, 100% 150px, 100% 100%, 0 100%);
      padding-top: 170px;
    }
</style>
