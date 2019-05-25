<template>
  <div class="demo flex-shrink-0">
    <div id="nav" class="text-center">
      <router-link to="/">Home</router-link> |
      <router-link to="/demo">Demo</router-link>
    </div>

    <div class="container pt-5 pt-lg-0">
      <loading :active.sync="isLoading" :can-cancel="false" :is-full-page="true"></loading>
      <div class="row pb-2">
        <div class="col-12">
          <p v-html="$t('explanation')"></p>

          <p>
            {{ $t('prepared_examples') }}
            <span v-for="(text, name) in text_variants" :key="name">
              <a href="javascript:void(0);" @click="setText(text)">{{name}}</a>,
            </span>
          </p>
        </div>
      </div>
      <div class="row">
        <div class="col-12">
          <div class="form-group">
            <textarea class="form-control" id="inputText" rows="7" v-model="input_text"></textarea>
          </div>
        </div>
      </div>
      <div class="row justify-content-end">
        <div class="col-12 col-lg-2">
          <button
            type="button"
            class="btn btn-primary btn-lg btn-block"
            @click="predictMentions"
            id="run"
          >{{ $t("run") }}</button>
        </div>
      </div>
      <div v-for="(prediction, idx) in predictions" :key="idx">
        <hr>
        <div class="container">
          <div class="row">
            <div class="col-12 col-lg-10">
              <Attention :weights="prediction.prediction.attention" :mention="prediction.mention"></Attention>
            </div>
            <div class="col-12 col-lg-2 pt-lg-0 pt-4" id="labels">
              <Labels :prediction="prediction.prediction.labels"></Labels>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import $backend from "../backend";
// Import component
import Loading from "vue-loading-overlay";
// Import stylesheet
import texts from "../demo_texts";

import Attention from "@/components/Attention.vue";
import Labels from "@/components/Labels.vue";

import { page, event } from 'vue-analytics'

export default {
  name: "demo",
  components: {
    Loading,
    Attention,
    Labels
  },
  data() {
    return {
      isLoading: false,
      text_variants: texts,
      input_text: "",
      error: "",
      predictions: []
    };
  },
  methods: {
    track () {
      page('/demo')
    },
    setText(text) {
      event('select_example')
      this.input_text = text;
    },
    predictMentions() {
      event('predict')
      this.isLoading = true;
      $backend
        .predictMentions(this.input_text)
        .then(responseData => {
          this.predictions = responseData;
          this.isLoading = false;
          this.$nextTick(function() {
            var element = document.getElementById("run");
            element.scrollIntoView({ behavior: "smooth", alignToTop: true });
          });
        })
        .catch(error => {
          this.error = error.message;
          this.isLoading = false;
        });
    }
  },
  mounted: function(){
    this.track()
  }
};
</script>
