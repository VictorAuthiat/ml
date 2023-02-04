require_relative "ml/utils"

module ML
  VERSION = "0.1.9"

  autoload :Gradient, "./lib/ml/gradient"
  autoload :LogLoss,  "./lib/ml/log_loss"
  autoload :Mapper,   "./lib/ml/mapper"
  autoload :Model,    "./lib/ml/model"
  autoload :Neuron,   "./lib/ml/neuron"
  autoload :Train,    "./lib/ml/train"
end
