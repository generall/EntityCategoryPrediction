<template>
  <div>
    <div class="container small m-0 p-0">
      <div class="row">
        <div class="col-12">
          <b>{{$t('labels')}}</b>
        </div>
      </div>

      <div class="row mt-2" v-for="value in sortPredictions(prediction)" :key="value.label">
        <div class="col-12">
          <a :href="'https://en.wikipedia.org/wiki/Category:' + value.label"> {{value.label}} </a>
          <div class="progress">
            <div
              class="progress-bar"
              :class="{
                'bg-warning': value.prob < 0.5,
                'bg-success': value.prob >= 0.5
              }"
              role="progressbar"
              :style="{width: value.prob * 100 + '%'}"
              :aria-valuenow="value.prob * 100"
              aria-valuemin="0"
              aria-valuemax="100"
            ></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>


<script>
export default {
  name: "Labels",
  props: {
    prediction: Object
  },
  methods: {
    sortPredictions(predictions) {
      var sorted_keys = Object.keys(predictions).sort(function(a, b) {return -(predictions[a] - predictions[b])})
      var res = []

      for(var x of sorted_keys) {
        res.push({
          label: x,
          prob: predictions[x]
        })
      }

      return res
    }
  }
};
</script>
