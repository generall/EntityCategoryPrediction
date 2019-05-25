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

import VueI18n from 'vue-i18n'
import messages from './translation.js'

Vue.use(VueI18n)

const i18n = new VueI18n({
  locale: 'en', // set locale
  messages, // set locale messages
})


new Vue({
  i18n,
  router,
  store,
  render: h => h(App)
}).$mount('#app')
