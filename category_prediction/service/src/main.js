import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'

import './filters'

import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'

Vue.config.productionTip = false

Vue.component('font-awesome-icon', FontAwesomeIcon)

import 'bootstrap'
import 'bootstrap/dist/css/bootstrap.min.css'
import 'vue-loading-overlay/dist/vue-loading.css';


new Vue({
  router,
  store,
  render: h => h(App)
}).$mount('#app')
