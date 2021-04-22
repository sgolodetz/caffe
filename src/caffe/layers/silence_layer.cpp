#include <vector>

#include "caffe/layers/silence_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SilenceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_set(bottom[i]->count(), Dtype(0),
                bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SilenceLayer);
#endif

INSTANTIATE_CLASS(SilenceLayer);
// BEGIN SMG
// See: https://stackoverflow.com/questions/42993870/caffe-layer-creation-failure-unknown-layer-type/43060851#43060851
//REGISTER_LAYER_CLASS(Silence);
// END SMG

}  // namespace caffe
