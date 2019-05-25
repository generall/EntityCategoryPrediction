<template>
  <div class="container small m-0 p-0">
    <div class="row mb-2">
      <div class="col-1">
        <b>{{ $t('layers') }}</b>
      </div>
      <div class="col-12 col-lg-11 text-center d-none d-lg-flex">
        <b>Attention - <span v-for="word in mention" class="mention" :key="word">{{word}} </span></b>
        <!-- : <span v-for="(head_color, idx) in head_colors" :key='head_color'>
          <font-awesome-icon icon="square-full" :style="{color: 'rgb(' + head_color + ')' }"/> - head {{idx + 1}} &nbsp;
        </span>-->
      </div>
    </div>
    <div class="row">
      <div class="col-12 col-lg-1 d-flex align-items-center pr-0">
        <ul class="nav nav-pills flex-row flex-lg-column">
          <li class="nav-item" v-for="layer_num in num_layers" :key="layer_num">
            <a
              class="nav-link p-2"
              :class="{active: layer_num === (selected_layer + 1)}"
              href="javascript:void(0)"
              @click="select_layer(layer_num - 1)"
            >
              <font-awesome-icon icon="layer-group"/> {{layer_num}}
            </a>
          </li>
        </ul>
      </div>

      <div class="col-11">
        <div class="row d-flex d-lg-none">
          <div class="col-12 pt-4">
            <b>Attention - <span v-for="word in mention" class="mention" :key="word">{{word}} </span></b>
          </div>
        </div>
        <div class="row mt-3" v-for="(tokens, idx) in weights" :key="idx">
          <div class="col-12 text-justify">
            <span
              v-for="(token, idx) in filter_tokens(tokens)"
              :key="idx"
              :class="{mention: mention.includes(token.token.toLowerCase())}"
            >
              <span :style="'background-color: ' + get_color(token.weights) + ';'">{{token.token}}</span> &shy;
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>


<script>
import { library } from "@fortawesome/fontawesome-svg-core";
import { faLayerGroup, faSquareFull } from "@fortawesome/free-solid-svg-icons";

library.add(faLayerGroup);
library.add(faSquareFull);

export default {
  name: "Attention",
  props: {
    weights: Object,
    mention: Array
  },
  methods: {
    filter_tokens(tokens) {
      return tokens.filter(token => !token.token.includes("@@"));
    },
    select_layer(new_layer) {
      this.selected_layer = new_layer;
    },
    get_color(weights) {
      // weights shape: [num_layers, num_heads]
      var layer_weights = weights[this.selected_layer];
      let weight = layer_weights.reduce((x, y) => x + y); //layer_weights.indexOf(Math.max(...layer_weights))
      let color = this.head_colors[2]; // [max_head % this.head_colors.length]
      var alpha = Math.min(weight, 0.8);
      if (alpha < 0.15) {
        alpha = 0;
      }
      // if (alpha > 0.7) {
      //   alpha = 0.8
      // } else if (alpha > 0.5) {
      //   alpha = 0.5
      // } else if (alpha > 0.2) {
      //   alpha = 0.3
      // } else {
      //   alpha = 0.0
      // }
      let style = "rgba(" + color + ", " + alpha + ")";
      return style;
    }
  },
  data: function() {
    return {
      num_layers: 3,
      selected_layer: 0,
      head_colors: [
        "26, 188, 156",
        "46, 204, 113",
        "52, 152, 219",
        "155, 89, 182",
        "22, 160, 133"
      ]
    };
  }
};
</script>


<style>
.context {
  font-size: 12px;
}

.mention {
  font-weight: bold;
  text-transform: capitalize;
}
</style>
