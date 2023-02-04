# frozen_string_literal: true

module ML
  class Train
    attr_reader :neuron, :w_data, :b_data, :score, :loss

    THRESHOLD = 0.5

    def self.call(neuron)
      new(neuron).call
    end

    def initialize(neuron)
      @neuron = neuron
      @w_data = nil
      @b_data = nil
      @score  = nil
      @loss   = []
    end

    def call
      neuron.n_iter.times { update_data! }

      @score = ML::Utils.accuracy_score(neuron.y_data, y_pred)
    end

    def y_pred
      @_y_pred ||= begin
        ML::Model
          .call(x_data, w_data, b_data)
          .map { |arr| [arr[0] >= THRESHOLD] }
      end
    end

    def mapping
      @_mapping ||= begin
        mapper = ML::Mapper.new(neuron.x_data)

        mapper.value
      end
    end

    def w_data
      @w_data ||= mapping[0]
    end

    def b_data
      @b_data ||= mapping[1]
    end

    def x_data
      neuron.x_data
    end

    private

    # TODO: refactor perhaps split in several methods
    def update_data!
      model_data       = build_model_data
      dw_data, db_data = ML::Gradient.call(model_data, x_data, neuron.y_data)
      dw_data_map      = dw_data.map { |arr| arr.map { |e| e * neuron.learning_rate } }

      @w_data = @w_data.map.with_index { |e, i| dw_data_map[i].map { |f| e[0] - f } }
      @b_data = [b_data[0] - neuron.learning_rate * db_data]
    end

    def build_model_data
      model_data = ML::Model.call(x_data, w_data, b_data)
      @loss << ML::LogLoss.call(model_data, neuron.y_data)

      model_data
    end
  end
end
